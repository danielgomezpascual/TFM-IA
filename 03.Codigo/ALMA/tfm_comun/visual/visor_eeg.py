"""
Visor de EEG interactivo para Jupyter: eliges partición, paciente y fichero, y ves su EEG con las crisis marcadas.

Motivo para cambiar: que cambien los controles del visor (qué se puede elegir y cómo se navega).
Cada pieza viene de su módulo: la lista de pacientes de dataset/catalogo.py, la lectura de dataset/edf.py
y dataset/anotaciones.py, el montaje de senal/montaje.py, el filtro de senal/filtrado.py y el dibujo de
visual/trazado_eeg.py. Aquí solo se conectan con controles de ipywidgets.

Debajo del trazado, dos paneles extra:
    - 📊 Frecuencias (casilla): espectro de la ventana, ritmo dominante y una cabeza por banda (panel_frecuencias.py).
    - 🧠 ¿Dónde se origina? (botón): cabeza con el aumento de potencia por electrodo al inicio de la crisis
      (panel_origen.py) y, si está el 3D, el cerebro con sLORETA, como en F01.03v (fuentes.py y cerebro.py).

Para que vaya rápido con cualquier fichero (hay EDF de varias horas):
    - No se lee el fichero entero: en cada dibujo se lee del disco SOLO la ventana que se ve
      (+ MARGEN_FILTRO_S antes, para que el filtro arranque fuera de la figura y no deje un eco al principio).
    - La duración y la frecuencia salen del inventario de F01.01, sin abrir el EDF.
    - La figura se manda al notebook como PNG a DPI_PANTALLA (menos peso que a la resolución por defecto).

Uso (en un notebook):
    VisorEEG().mostrar()
"""

import importlib.util
import io

import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output, display

from tfm_comun.dataset.anotaciones import crisis_del_fichero
from tfm_comun.dataset.catalogo import ficheros_paciente, leer_inventario, resumen_pacientes
from tfm_comun.dataset.edf import leer_edf_tusz
from tfm_comun.dataset.particiones import SPLITS
from tfm_comun.entorno.rutas import RUTA_MODELOS_CABEZA, buscar_edf
from tfm_comun.senal.aumento_electrodos import aumento_por_electrodo, electrodos_planos
from tfm_comun.senal.espectro import espectro
from tfm_comun.senal.filtrado import PASO_BANDA_HZ, filtrar
from tfm_comun.senal.fuentes import ELECTRODOS_EXCLUIDOS
from tfm_comun.senal.montaje import montaje_tcp
from tfm_comun.senal.tramos_origen import recortar_inicio_plano, tramos_origen
from tfm_comun.visual.cabeza import POSICIONES_1020
from tfm_comun.visual.iconos import ICONO_SPLIT, icono_etiqueta
from tfm_comun.visual.panel_frecuencias import figura_frecuencias
from tfm_comun.visual.panel_origen import figura_aumento_electrodos
from tfm_comun.visual.trazado_eeg import dibujar_trazado

VENTANAS_S = [10, 20, 30, 60]          # segundos que se ven a la vez
ESCALAS_UV = [20, 50, 100, 200, 500]   # µV entre canales
SEGUNDOS_ANTES_CRISIS = 5              # al saltar a una crisis, cuántos segundos de antes se ven
MARGEN_FILTRO_S = 10                   # segundos que se leen antes de la ventana para que el filtro se asiente
DPI_PANTALLA = 80                      # resolución de la imagen que se manda al notebook
HAY_3D = importlib.util.find_spec('pyvista') is not None   # el cerebro 3D necesita pyvista (como en F01.03v)


class VisorEEG:
    """Controles + figura. Cada vez que cambia algo se lee del disco solo la ventana visible y se dibuja."""

    def __init__(self):
        self.inventario = leer_inventario()
        self.ruta, self.fs, self.duracion_s, self.crisis = None, None, None, None

        self.split = w.ToggleButtons(options=[(f'{ICONO_SPLIT[s]} {s}', s) for s in SPLITS], value='dev',
                                     description='Partición')
        self.solo_con_crisis = w.Checkbox(value=True, description='Solo pacientes con crisis', indent=False)
        self.paciente = w.Dropdown(description='Paciente', layout=w.Layout(width='620px'))
        self.fichero = w.Dropdown(description='Fichero', layout=w.Layout(width='620px'))
        self.ir_a_crisis = w.Dropdown(description='Ir a crisis', layout=w.Layout(width='620px'))
        self.inicio = w.FloatSlider(description='Inicio (s)', step=1, continuous_update=False, readout_format='.0f',
                                    layout=w.Layout(width='620px'))
        self.atras = w.Button(description='◀', layout=w.Layout(width='50px'), tooltip='Ventana anterior')
        self.adelante = w.Button(description='▶', layout=w.Layout(width='50px'), tooltip='Ventana siguiente')
        self.ventana = w.Dropdown(options=VENTANAS_S, value=20, description='Ventana (s)')
        self.escala = w.Dropdown(options=ESCALAS_UV, value=100, description='Escala (µV)')
        self.filtro = w.Checkbox(value=True, indent=False,
                                 description=f'Filtro {PASO_BANDA_HZ[0]}-{PASO_BANDA_HZ[1]} Hz + notch')
        self.ver_frecuencias = w.Checkbox(value=True, indent=False, description='📊 Frecuencias y cabezas por banda')
        self.boton_origen = w.Button(description='🧠 ¿Dónde se origina esta crisis?', layout=w.Layout(width='280px'),
                                     tooltip='La crisis elegida en "Ir a crisis" o, si no hay, la que se ve')
        self.info = w.HTML()
        self.salida = w.Output()
        self.salida_origen = w.Output()

        self.split.observe(lambda _: self._rellenar_pacientes(), 'value')
        self.solo_con_crisis.observe(lambda _: self._rellenar_pacientes(), 'value')
        self.paciente.observe(lambda _: self._rellenar_ficheros(), 'value')
        self.fichero.observe(lambda _: self._abrir_fichero(), 'value')
        self.ir_a_crisis.observe(lambda _: self._saltar_a_crisis(), 'value')
        for control in (self.inicio, self.ventana, self.escala, self.filtro, self.ver_frecuencias):
            control.observe(lambda _: self._dibujar(), 'value')
        self.atras.on_click(lambda _: self._mover(-1))
        self.adelante.on_click(lambda _: self._mover(+1))
        self.boton_origen.on_click(lambda _: self._mostrar_origen())

    def mostrar(self):
        """Pinta los controles y la figura en el notebook, y abre el primer paciente de 🟡 dev."""
        display(w.VBox([
            w.HBox([self.split, self.solo_con_crisis]),
            self.paciente, self.fichero, self.ir_a_crisis,
            w.HBox([self.atras, self.inicio, self.adelante]),
            w.HBox([self.ventana, self.escala, self.filtro]),
            w.HBox([self.ver_frecuencias, self.boton_origen]),
            self.info, self.salida, self.salida_origen,
        ]))
        self._rellenar_pacientes()

    # --- Listas de los desplegables -------------------------------------------------------------------------

    def _rellenar_pacientes(self):
        """Pacientes de la partición, de más a menos crisis: 'aaaaaajy · 12 crisis · 4 ficheros · 1.3 h · 🟥 fnsz (♦️)'."""
        pacientes = resumen_pacientes(self.inventario, self.split.value)
        if self.solo_con_crisis.value:
            pacientes = pacientes[pacientes.n_crisis > 0]
        self.paciente.options = [
            (f'{p} · {f.n_crisis} crisis · {f.n_ficheros} ficheros · {f.horas:.1f} h · '
             + ' '.join(icono_etiqueta(t) for t in f.tipos), p)
            for p, f in pacientes.iterrows()]

    def _rellenar_ficheros(self):
        """EDF del paciente: 's001_t000 · 02_tcp_le · 29 min · 2 crisis (🟥 fnsz (♦️))'."""
        if self.paciente.value is None:
            self.fichero.options = []
            return
        ficheros = ficheros_paciente(self.inventario, self.paciente.value)
        self.fichero.options = [
            (f'{f.session}_{f.token} · {f.montage} · {f.edf_dur_s / 60:.0f} min · {f.n_seiz_events_bi} crisis'
             + (f" ({' '.join(icono_etiqueta(t) for t in f.labels.split(';'))})" if f.labels else ''), f.file)
            for f in ficheros.itertuples()]

    # --- Fichero y navegación -------------------------------------------------------------------------------

    def _abrir_fichero(self):
        """Busca el EDF y lee sus crisis (sin leer la señal). Duración y frecuencia salen del inventario."""
        if self.fichero.value is None:
            return
        self.salida_origen.clear_output()
        fila = self.inventario[self.inventario.file == self.fichero.value].iloc[0]
        # Dentro de "with self.salida", si algo falla el error se ve debajo del visor (si no, se perdería)
        with self.salida:
            self.ruta = buscar_edf(self.fichero.value)
            self.crisis = crisis_del_fichero(self.ruta)
        self.fs, self.duracion_s = fila.fs, fila.edf_dur_s

        self.ir_a_crisis.options = [('—', None)] + [
            (f'{i + 1} · {icono_etiqueta(c.tipo)} · {c.inicio_s:.0f} s → {c.fin_s:.0f} s ({c.duracion_s:.0f} s)', i)
            for i, c in enumerate(self.crisis.itertuples())]
        self.info.value = (f'<b>{self.fichero.value}</b> · {self.duracion_s / 60:.1f} min · {self.fs:.0f} Hz · '
                           f'{fila.montage} · {len(self.crisis)} crisis')
        primera_crisis = self.crisis.inicio_s.min() - SEGUNDOS_ANTES_CRISIS if len(self.crisis) else 0
        nuevo_inicio = float(round(max(0, primera_crisis)))
        with self.inicio.hold_trait_notifications():   # cambiar max y value sin dibujar dos veces
            self.inicio.max = max(0, self.duracion_s - self.ventana.value)
            cambia = self.inicio.value != nuevo_inicio
            self.inicio.value = nuevo_inicio
        if not cambia:
            self._dibujar()

    def _saltar_a_crisis(self):
        """Pone el inicio unos segundos antes de la crisis elegida."""
        if self.ir_a_crisis.value is not None:
            self.inicio.value = float(round(max(0, self.crisis.inicio_s.iloc[self.ir_a_crisis.value]
                                                - SEGUNDOS_ANTES_CRISIS)))

    def _mover(self, sentido):
        """Avanza (+1) o retrocede (-1) una ventana entera."""
        self.inicio.value = min(self.inicio.max, max(0, self.inicio.value + sentido * self.ventana.value))

    # --- Lectura del trozo y dibujo -------------------------------------------------------------------------

    def _leer_ventana(self, inicio_s, fin_s):
        """Lee del disco solo [inicio - margen, fin], lo pasa a TCP en µV, filtra si toca y quita el margen.

        Devuelve (senal_uv, canales, fs, inicio_real_s, raw): la matriz TCP de la ventana, el segundo en que
        empieza y el Raw referencial leído (con el margen), para el panel de frecuencias.
        """
        raw = leer_edf_tusz(self.ruta, inicio_s - MARGEN_FILTRO_S, fin_s)
        tcp = montaje_tcp(raw)
        fs = tcp.info['sfreq']
        senal = tcp.get_data() * 1e6
        if self.filtro.value:
            senal = filtrar(senal, fs)
        leido_desde_s = tcp.first_samp / fs
        quitar = max(0, round((inicio_s - leido_desde_s) * fs))
        return senal[:, quitar:], tcp.ch_names, fs, leido_desde_s + quitar / fs, raw

    def _dibujar(self):
        """Lee la ventana elegida, la dibuja con trazado_eeg.dibujar_trazado() y la manda al notebook como PNG."""
        if self.ruta is None:
            return
        self.inicio.max = max(0, self.duracion_s - self.ventana.value)
        with self.salida:
            senal, canales, fs, desde_s, raw = self._leer_ventana(self.inicio.value, self.inicio.value + self.ventana.value)
            fig = dibujar_trazado(senal, canales, fs, desde_s, self.crisis, escala_uv=self.escala.value,
                                  titulo=f'{self.fichero.value} · paciente {self.paciente.value} · {self.split.value}')
            imagenes = [_a_png(fig)]
            if self.ver_frecuencias.value:
                imagenes.append(_a_png(self._figura_frecuencias(raw, desde_s)))
            clear_output(wait=True)
            for imagen in imagenes:
                display(imagen)

    def _figura_frecuencias(self, raw, desde_s):
        """Panel de frecuencias de la ventana: electrodos 10-20 con referencia promedio, filtrados, sin el margen."""
        electrodos = [c for c in raw.ch_names if c in POSICIONES_1020 and c not in (*ELECTRODOS_EXCLUIDOS, 'T1', 'T2')]
        fs = raw.info['sfreq']
        datos = raw.get_data(picks=electrodos) * 1e6
        datos = filtrar(datos - datos.mean(axis=0), fs)[:, round((desde_s - raw.first_samp / fs) * fs):]
        frecuencias, potencia = espectro(datos, fs)
        return figura_frecuencias(frecuencias, potencia, electrodos,
                                  f'Frecuencias de la ventana ({len(electrodos)} electrodos, referencia promedio)')

    # --- ¿Dónde se origina la crisis? ---------------------------------------------------------------------

    def _crisis_elegida(self):
        """La crisis de "Ir a crisis" o, si no hay ninguna elegida, la que más se ve en la ventana."""
        if self.ir_a_crisis.value is not None:
            return self.ir_a_crisis.value
        desde, hasta = self.inicio.value, self.inicio.value + self.ventana.value
        solape = (self.crisis.fin_s.clip(upper=hasta) - self.crisis.inicio_s.clip(lower=desde)).clip(lower=0)
        return int(solape.values.argmax()) if len(solape) and solape.max() > 0 else None

    def _mostrar_origen(self):
        """Cabeza con el aumento por electrodo (al momento) y, si se puede, el cerebro 3D con sLORETA."""
        with self.salida_origen:
            clear_output()
            indice = None if self.crisis is None else self._crisis_elegida()
            if indice is None:
                print('Elige una crisis en "Ir a crisis" o muévete hasta una (zona sombreada).')
                return
            c = self.crisis.iloc[indice]
            tramo_inicio, tramo_basal, nota = tramos_origen(self.crisis, indice, self.duracion_s)
            if tramo_basal is None:
                print(f'🔴 No se puede comparar: {nota}.')
                return
            print('Calculando...')

            # Se lee solo lo que va del primer tramo al último; los tramos pasan a segundos de lo leído
            raw = leer_edf_tusz(self.ruta, min(tramo_basal[0], tramo_inicio[0]), max(tramo_basal[1], tramo_inicio[1]))
            leido_desde_s = raw.first_samp / raw.info['sfreq']
            tramo_basal = recortar_inicio_plano(raw.get_data(picks='eeg') * 1e6, raw.info['sfreq'], tramo_basal,
                                                leido_desde_s)
            basal = (tramo_basal[0] - leido_desde_s, tramo_basal[1] - leido_desde_s)
            inicio = (tramo_inicio[0] - leido_desde_s, tramo_inicio[1] - leido_desde_s)
            planos = electrodos_planos(raw, basal)
            aumento = aumento_por_electrodo(raw, basal, inicio, excluir=planos)

            subtitulo = (f'Crisis {indice + 1} ({c.tipo}) · inicio {tramo_inicio[0]:.0f}-{tramo_inicio[1]:.0f} s frente a '
                         f'basal {tramo_basal[0]:.0f}-{tramo_basal[1]:.0f} s ({nota})'
                         + (f' · sin {", ".join(planos)} (planos)' if planos else ''))
            clear_output(wait=True)
            print(f'Crisis {indice + 1}: {icono_etiqueta(c.tipo)} · los electrodos que más crecen: '
                  + ', '.join(f'{e} ({v:+.1f} dB)' for e, v in aumento.head(3).items()))
            if aumento.max() <= 1:
                print('🔴 Ningún electrodo crece respecto al basal: el basal probablemente no estaba tranquilo '
                      '(otra crisis cerca, artefactos). Tómalo con cautela.')
            display(_a_png(figura_aumento_electrodos(aumento, '¿Dónde se origina la crisis?', subtitulo)))
            if HAY_3D:
                print('Calculando el cerebro 3D con sLORETA (la primera vez puede tardar un poco)...')
                self._cerebro_3d(raw, basal, inicio, planos, subtitulo)
            else:
                print('🔴 Sin pyvista no se dibuja el cerebro 3D (solo la cabeza).')

    def _cerebro_3d(self, raw, basal, inicio, planos, subtitulo):
        """El cerebro visto desde arriba con el origen estimado, con las funciones de F01.03v."""
        from tfm_comun.senal.fuentes import origen_crisis, resumen_por_region
        from tfm_comun.visual.cerebro import VISTAS, capturas_cerebro, figura_desde_arriba, limites_color
        try:
            origen = origen_crisis(raw, basal, inicio, excluir=planos, carpeta_modelos=RUTA_MODELOS_CABEZA)
            limites = limites_color(origen['cociente_db'])
            maximo, lobulos = resumen_por_region(origen['cociente_db'], origen['vertices'], origen['subjects_dir'],
                                                 limites['fmin'])
            captura = capturas_cerebro(origen, limites, vistas={'arriba': VISTAS['arriba']})['arriba']
            fig = figura_desde_arriba(origen, captura, limites, '¿Dónde se origina la crisis? Vista desde arriba',
                                      subtitulo)
        except Exception as error:   # el 3D es un extra: si falla, se dice por qué y la cabeza ya está dibujada
            print(f'🔴 No se ha podido calcular el cerebro 3D: {error}')
            return
        print(f'Origen estimado (sLORETA): {maximo["región"]} · lóbulo {maximo["lóbulo"]} · '
              f'hemisferio {maximo["hemisferio"]} · +{maximo["aumento (dB)"]:.1f} dB. '
              'Orientativo: cerebro plantilla y ~19 electrodos (precisión de lóbulo).')
        display(_a_png(fig))
        display(lobulos.head(5))


def _a_png(fig):
    """Convierte una figura en una imagen PNG a DPI_PANTALLA (pesa poco y se pinta rápido) y la cierra."""
    imagen = io.BytesIO()
    fig.savefig(imagen, format='png', dpi=DPI_PANTALLA)
    plt.close(fig)
    return Image(imagen.getvalue())
