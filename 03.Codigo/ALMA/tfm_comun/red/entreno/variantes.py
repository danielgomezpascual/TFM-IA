"""
Entrenador de variantes: entrenar una variante de red con todas sus semillas y rondas, o leerla del registro.

Motivo para cambiar: que cambie cómo se repite el entrenamiento de una variante (semillas, rondas, qué se guarda).

Imagen: fase_03/img/entrenar_una_variante.png (de la configuración a las puntuaciones, pasando por el registro)

Por qué una clase
=================
F03.05, F03.06, F03.07 y F03.08 llevaban cada uno su copia de la misma función (entrenar_configuracion o
entrenar_variante), con pequeñas diferencias: una guardaba los entrenamientos sueltos y otra no, una usaba puntos de
control y otra no... Copiar código hace que un arreglo en un notebook no llegue a los demás. Aquí está UNA vez.

La clase recibe al crearse todo lo que NO cambia entre variantes (inyección de dependencias): los datos de la red,
el dispositivo, el registro, las semillas y dónde se guardan los modelos. Después, entrenar(nombre, config) solo
necesita lo que SÍ cambia: la configuración de esa variante.

Qué hace entrenar(nombre, config), paso a paso
==============================================
    1. Calcula la HUELLA de la configuración (red/experimentos/huella.py) junto con el conjunto, el modo prueba, las
       semillas y (si se da) el notebook. Si cualquier cosa cambia, cambia la huella.
    2. Si la huella ya está en el registro con las mismas ventanas de validación, lee sus puntuaciones y no entrena.
    3. Si no, para cada semilla y cada ronda de validación:
        a. si ese entrenamiento suelto ya se hizo en una ejecución cortada, lo lee (registro.parciales);
        b. si no, fija la semilla, crea la red (y, si se da, la prepara: p. ej. copiar los pesos preentrenados de
           F03.07), la entrena con entrenar_red (con punto de control: si se corta, sigue desde la última época),
           la puntúa en su ronda de validación y guarda el modelo (.pt) y el entrenamiento suelto.
    4. Devuelve ResultadoVariante: la huella, las puntuaciones (semillas x filas), las curvas, de dónde ha salido
       ('del registro' o 'entrenado') y los minutos.
El notebook se encarga después de medir, comparar y apuntar el experimento en el registro (registro.guardar).
"""

import time
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from tfm_comun.entorno.nombres_fichero import nombre_fichero
from tfm_comun.red.entreno.entrenar_red import entrenar_red
from tfm_comun.red.entreno.puntuacion import puntuar_red
from tfm_comun.red.experimentos.huella import huella
from tfm_comun.red.lotes.orden_epoca import grupos_espaciados, peso_por_paciente
from tfm_comun.red.modelo import crear_red
from tfm_comun.red.reproducibilidad import fijar_semillas


class ResultadoVariante(NamedTuple):
    """Lo que sale de entrenar una variante. Se puede desempaquetar: h, P, historial, estado, minutos = resultado."""
    huella: str              # las 10 letras que identifican la configuración
    puntuaciones: np.ndarray  # (semillas, filas): probabilidad de ⏳ de cada semilla; NaN fuera de validación
    historial: pd.DataFrame  # las curvas de aprendizaje de cada semilla y ronda
    estado: str              # 'del registro' o 'entrenado'
    minutos: float           # lo que ha tardado (0 si se ha leído)


class EntrenadorVariantes:
    """Entrena (o lee del registro) variantes de red, todas con los mismos datos, semillas y rondas.

    Parámetros (lo que es igual para todas las variantes de un notebook):
        red_datos: lo que devuelve red/preparacion.py: preparar_datos_red (filas, pacientes, rondas, validar, datos...).
        dispositivo: torch.device donde se entrena.
        registro: Registro (red/experimentos/registro.py) de la carpeta del notebook.
        carpeta_modelos: dónde se guarda cada red entrenada (<huella>_semilla<n>_<ronda>.pt).
        semillas: lista de semillas (p. ej. [42, 43, 44]).
        conjunto, modo_prueba: entran en la huella (otro conjunto u otro modo = otro experimento).
        notebook: None o el código del notebook ('F03.06'). Entra en la huella SOLO si se da: F03.05 no lo ponía y así
            sus experimentos ya registrados conservan su huella.
        reentrenar: True = no leer del registro y borrar lo que hubiera a medias de esa huella.
        carpeta_puntos_control: dónde se guardan los puntos de control (None = sin puntos de control).
        lote_evaluacion: filas por lote al medir (solo cambia la memoria de la GPU; la DCRNN necesita lotes pequeños).
        barra: función (iterable, descripción) -> iterable con barra de progreso (p. ej. con tqdm), o None.
        nombre_en_modelo: con qué clave se guarda el nombre de la variante en el .pt ('variante'; F03.05 usaba 'peldaño').
    """

    def __init__(self, red_datos, dispositivo, registro, carpeta_modelos, semillas, conjunto, modo_prueba,
                 notebook=None, reentrenar=False, carpeta_puntos_control=None, lote_evaluacion=1024, barra=None,
                 nombre_en_modelo='variante'):
        self.red_datos, self.dispositivo, self.registro = red_datos, dispositivo, registro
        self.carpeta_modelos = carpeta_modelos
        self.carpeta_modelos.mkdir(parents=True, exist_ok=True)
        self.semillas, self.conjunto, self.modo_prueba, self.notebook = list(semillas), conjunto, modo_prueba, notebook
        self.reentrenar, self.carpeta_puntos_control = reentrenar, carpeta_puntos_control
        self.lote_evaluacion, self.barra, self.nombre_en_modelo = lote_evaluacion, barra, nombre_en_modelo
        self.ids = red_datos.filas.id_ventana.to_numpy()
        self.ids_validacion = self.ids[red_datos.validar]

    def huella(self, config):
        """La huella de una configuración en este notebook (la misma que se usa en el registro)."""
        extra = {'conjunto': self.conjunto, 'modo_prueba': self.modo_prueba, 'semillas': self.semillas}
        if self.notebook is not None:
            extra['notebook'] = self.notebook
        return huella({**config, **extra})

    def entrenar(self, nombre, config, preparar_red=None):
        """Entrena (o lee) una variante. Devuelve un ResultadoVariante.

        Parámetros:
            nombre: el nombre de la variante (para las barras y el .pt).
            config: su configuración. Claves que se usan: 'arquitectura', 'entrenamiento' (ajustes de entrenar_red),
                'k' (ventanas de la secuencia), 'separacion_s' (0/None = sin grupos espaciados) y, opcional,
                'mismo_peso' (True = cada paciente pesa lo mismo en cada época).
            preparar_red: función (red) -> None que se aplica a la red recién creada, antes de entrenar
                (p. ej. en F03.07: copiar los pesos preentrenados y congelar el codificador).
        """
        d = self.red_datos
        h = self.huella(config)
        P = np.full((len(self.semillas), len(d.filas)), np.nan)
        guardado = None if self.reentrenar else self.registro.buscar(h, self.ids_validacion)
        if guardado is not None:                                       # paso 2: ya estaba en el registro
            P[:, d.validar] = guardado[0]
            return ResultadoVariante(h, P, guardado[1], 'del registro', 0.0)
        if self.reentrenar:                                            # empezar de cero: fuera lo que hubiera a medias
            self.registro.parciales.borrar(h)
            if self.carpeta_puntos_control is not None and self.carpeta_puntos_control.exists():
                for ruta in self.carpeta_puntos_control.glob(f'{h}_*'):
                    ruta.unlink()

        inicio, historiales = time.time(), []
        separacion = config.get('separacion_s')
        grupos_todos = grupos_espaciados(d.filas.ruta, d.filas.muestra_inicio, separacion, d.fs) if separacion else None
        for s, semilla in enumerate(self.semillas):                    # paso 3
            for ronda, entrena, valida in d.rondas:
                pos_e, pos_v = np.where(entrena)[0], np.where(valida)[0]
                parcial = self.registro.parciales.buscar(h, semilla, ronda, self.ids[pos_v])
                if parcial is not None:                                # 3a: ya acabó en una ejecución anterior
                    P[s, pos_v] = parcial[0]
                    historiales.append(parcial[1])
                    print(f'{nombre} · semilla {semilla} · {ronda}: ya estaba hecha (se lee)')
                    continue
                historial = self._entrenar_una(nombre, config, h, semilla, ronda, pos_e, pos_v, grupos_todos, preparar_red, P[s])
                self.registro.parciales.guardar(h, semilla, ronda, P[s, pos_v], self.ids[pos_v], historial)
                historiales.append(historial)
        return ResultadoVariante(h, P, pd.concat(historiales, ignore_index=True), 'entrenado', (time.time() - inicio) / 60)

    def _entrenar_una(self, nombre, config, h, semilla, ronda, pos_e, pos_v, grupos_todos, preparar_red, fila_P):
        """3b: una semilla en una ronda. Rellena fila_P en las filas de validación y devuelve su historial."""
        d = self.red_datos
        punto = None
        if self.carpeta_puntos_control is not None:
            punto = self.carpeta_puntos_control / f'{h}_semilla{semilla}_{nombre_fichero(ronda)}.pt'
            if punto.exists():
                print(f'{nombre} · semilla {semilla} · {ronda}: sigue desde su último punto de control')
        fijar_semillas(semilla)
        red = crear_red(d.n_canales, d.n_frecuencias, config['arquitectura']).to(self.dispositivo)
        if preparar_red is not None:
            preparar_red(red)
        descripcion = f'{nombre} · semilla {semilla} · {ronda}'
        red, historial, mejor = entrenar_red(
            red, d.datos, pos_e, pos_v, k=config['k'], semilla=semilla, pacientes_valida=d.pacientes[pos_v],
            pesos=peso_por_paciente(d.pacientes[pos_e]) if config.get('mismo_peso') else None,
            grupos=None if grupos_todos is None else grupos_todos[pos_e], ajustes=config['entrenamiento'],
            lote_evaluacion=self.lote_evaluacion, punto_control=punto,
            progreso=(lambda it: self.barra(it, descripcion)) if self.barra else None)
        fila_P[pos_v] = puntuar_red(red, d.datos, pos_v, k=config['k'], tamano_lote=self.lote_evaluacion)
        torch.save({'estado': red.state_dict(), 'configuracion': config, 'n_canales': d.n_canales,
                    'n_frecuencias': d.n_frecuencias, 'frecuencias_hz': d.frecuencias.tolist(),
                    'normalizacion': d.normalizacion, 'mejor_epoca': mejor['epoca'],
                    'auc_validacion': mejor['auc_validacion'], 'conjunto': self.conjunto, self.nombre_en_modelo: nombre},
                   self.carpeta_modelos / f'{h}_semilla{semilla}_{nombre_fichero(ronda)}.pt')
        if punto is not None:
            punto.unlink(missing_ok=True)                              # ya está apuntada: el punto de control sobra
        return historial.reset_index().assign(semilla=semilla, ronda=ronda)
