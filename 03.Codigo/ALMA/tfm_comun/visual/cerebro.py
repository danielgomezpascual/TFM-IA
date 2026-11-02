"""
Figuras del cerebro: dónde se origina la crisis, en 3D y vista desde arriba.

Motivo para cambiar: que cambie cómo se dibujan el cerebro 3D o el mapa del cuero cabelludo.
Los datos los calcula tfm_comun/senal/fuentes.py (origen_crisis); aquí solo se pintan.

El 3D lo dibuja pyvista directamente (pip install pyvista), sin ventana (fuera de pantalla): se lee
la superficie del cerebro plantilla, se pinta el aumento encima, se hacen capturas desde varios ángulos
y se montan en una figura de matplotlib. Así la figura se guarda como PNG y funciona igual al lanzar el
notebook con papermill. (El visor 3D de MNE, con Qt, daba capturas en negro en Windows.)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import mne

from tfm_comun.visual.tipografia import ESTILO_TIMES

SUPERFICIE = 'pial'         # 'pial' = cerebro tal cual; 'inflated' = inflado (se ven los surcos)
FONDO = 'white'
GRIS_CORTEZA = '#c9c9c9'    # lo que no supera el umbral (en el cerebro y en la cabeza)

# Escala de color: amarillo -> naranja -> ROJO OSCURO = máximo aumento (el foco se ve en rojo).
# Es la escala 'YlOrRd' sin su tramo inicial casi blanco, para que no se confunda con el gris;
# lo que queda por debajo del umbral se pinta gris (color 'under')
MAPA_COLOR = LinearSegmentedColormap.from_list(
    'amarillo_rojo', plt.get_cmap('YlOrRd')(np.linspace(0.25, 1, 256))).with_extremes(under=GRIS_CORTEZA)

# Vistas del cerebro 3D: nombre en la figura -> (desde dónde se mira, qué queda arriba en la imagen).
# Coordenadas del cerebro (RAS): x hacia la derecha, y hacia delante (nariz), z hacia arriba
VISTAS = {
    'izquierda': {'direccion': (-1, 0, 0), 'arriba': (0, 0, 1)},
    'derecha': {'direccion': (1, 0, 0), 'arriba': (0, 0, 1)},
    'delante': {'direccion': (0, 1, 0), 'arriba': (0, 0, 1)},
    'detrás': {'direccion': (0, -1, 0), 'arriba': (0, 0, 1)},
    'arriba': {'direccion': (0, 0, 1), 'arriba': (0, 1, 0)},     # desde arriba, con la nariz hacia arriba
}

ROJO_OSCURO = '#8b0000'


def limites_color(cociente_db, fraccion_umbral=0.6):
    """Límites de la escala de color del cerebro, relativos al máximo aumento.

    Solo se pinta lo que crece al menos fraccion_umbral (60 %) del máximo; lo demás queda gris.
    Así, en una crisis focal se enciende una zona pequeña y en una generalizada casi todo:
    el tamaño de la mancha dice si la crisis está localizada o repartida.
    El máximo es el percentil 99,5 (no el valor más alto) para que un punto aislado no mande.
    """
    maximo = max(float(np.percentile(cociente_db, 99.5)), 0.1)
    umbral = fraccion_umbral * maximo
    return {'fmin': umbral, 'fmid': (umbral + maximo) / 2, 'fmax': maximo}


def _recortar_blanco(imagen, margen=6):
    """Quita los bordes blancos de una captura (filas y columnas sin cerebro)."""
    con_algo = (imagen[..., :3] < 250).any(axis=2)
    filas, columnas = np.flatnonzero(con_algo.any(axis=1)), np.flatnonzero(con_algo.any(axis=0))
    if len(filas) == 0:
        return imagen
    return imagen[max(filas[0] - margen, 0):filas[-1] + margen, max(columnas[0] - margen, 0):columnas[-1] + margen]


def _superficies(origen, vecinos=6):
    """Superficie de cada hemisferio del cerebro plantilla con el aumento (dB) en cada uno de sus vértices.

    El aumento se calculó en ~10.000 puntos por hemisferio; la superficie tiene ~160.000 vértices. Cada
    vértice toma la media de los `vecinos` puntos calculados más cercanos, pesada por la distancia
    (más cerca, más peso), para que la mancha salga suave.
    Devuelve una lista de mallas de pyvista, con el aumento en la propiedad 'aumento_db'.
    """
    import pyvista
    from scipy.spatial import cKDTree

    carpeta = origen['subjects_dir'] / 'fsaverage' / 'surf'
    n_izquierdo = len(origen['vertices'][0])
    valores = [origen['cociente_db'][:n_izquierdo], origen['cociente_db'][n_izquierdo:]]
    mallas = []
    for hemi, vertices, valor in zip(('lh', 'rh'), origen['vertices'], valores):
        puntos, triangulos = mne.read_surface(carpeta / f'{hemi}.{SUPERFICIE}', verbose='error')
        distancia, cercanos = cKDTree(puntos[vertices]).query(puntos, k=vecinos)
        peso = 1 / np.maximum(distancia, 1e-3)
        caras = np.column_stack([np.full(len(triangulos), 3), triangulos]).ravel()
        malla = pyvista.PolyData(puntos, caras)
        malla['aumento_db'] = (valor[cercanos] * peso).sum(axis=1) / peso.sum(axis=1)
        mallas.append(malla)
    return mallas


def _escena(origen, limites, fuera_de_pantalla, tamano):
    """Escena de pyvista con los dos hemisferios: gris donde no se supera el umbral, de rojo a amarillo encima."""
    import pyvista
    escena = pyvista.Plotter(off_screen=fuera_de_pantalla, window_size=list(tamano))
    escena.set_background(FONDO)
    for malla in _superficies(origen):
        escena.add_mesh(malla, scalars='aumento_db', cmap=MAPA_COLOR, clim=[limites['fmin'], limites['fmax']],
                        below_color=GRIS_CORTEZA, smooth_shading=True, specular=0.2, ambient=0.25,
                        show_scalar_bar=False)
    return escena


def capturas_cerebro(origen, limites, vistas=VISTAS, tamano=(900, 750)):
    """Renderiza el cerebro 3D sin ventana y devuelve una captura (imagen RGB) por vista.

    Parámetros:
        origen: el diccionario de fuentes.origen_crisis().
        limites: los de limites_color().
        vistas: {nombre: {'direccion': (x, y, z), 'arriba': (x, y, z)}}. Por defecto, VISTAS.
    Devuelve: {nombre de la vista: imagen recortada}.
    """
    escena = _escena(origen, limites, fuera_de_pantalla=True, tamano=tamano)
    capturas = {}
    for nombre, vista in vistas.items():
        escena.view_vector(vista['direccion'], viewup=vista['arriba'])
        escena.reset_camera()
        capturas[nombre] = _recortar_blanco(escena.screenshot(return_img=True))
    escena.close()
    if all(c.mean() < 10 for c in capturas.values()):
        print('🔴 Las capturas del 3D han salido en negro: el render fuera de pantalla no funciona en este PC.')
    return capturas


def abrir_cerebro_interactivo(origen, limites):
    """Cerebro 3D para girarlo con el ratón (solo a mano, no con papermill).

    Si está instalado trame-pyvista (lo que pyvista usa para Jupyter), el cerebro se gira DENTRO del
    notebook. Si no, se abre en una ventana aparte y la celda espera hasta que se cierra.
    Si se acaba de instalar, hay que reiniciar el kernel: pyvista busca sus motores una vez al arrancar.
    """
    import importlib.util
    escena = _escena(origen, limites, fuera_de_pantalla=False, tamano=(1000, 800))
    escena.view_vector(VISTAS['izquierda']['direccion'], viewup=VISTAS['izquierda']['arriba'])
    dentro_del_notebook = importlib.util.find_spec('trame_pyvista') is not None
    escena.show(jupyter_backend='trame' if dentro_del_notebook else 'none')


def _barra_color(fig, ax, limites, texto):
    """Barra de color del aumento en dB, a juego con el cerebro. El triángulo gris de abajo = por debajo del umbral."""
    barra = fig.colorbar(ScalarMappable(Normalize(limites['fmin'], limites['fmax']), cmap=MAPA_COLOR),
                         ax=ax, fraction=0.035, pad=0.02, extend='min')    # triángulo gris = bajo el umbral
    barra.set_label(texto)
    return barra


def _titulo_rojo_negro(fig, titulo, subtitulo):
    """Cabecera de las figuras descriptivas del TFM: título en negro y una línea roja debajo."""
    fig.suptitle(titulo, fontsize=17, fontweight='bold', y=0.995)
    fig.text(0.5, 0.945, subtitulo, ha='center', fontsize=11.5, color='dimgray')
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.935, 0.935], color=ROJO_OSCURO, linewidth=2,
                              transform=fig.transFigure))


def figura_cerebro_3d(capturas, limites, titulo, subtitulo, region_maxima=None):
    """Figura con el cerebro 3D visto desde los cuatro lados (izquierda, derecha, delante, detrás).

    region_maxima: texto con la región del máximo, que se escribe al pie (p. ej. 'giro temporal superior derecho').
    """
    vistas = [v for v in ('izquierda', 'derecha', 'delante', 'detrás') if v in capturas]
    with plt.rc_context(ESTILO_TIMES):
        fig, ejes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, vista in zip(ejes.flat, vistas):
            ax.imshow(capturas[vista])
            ax.set_title(f'Vista desde {"la " if vista in ("izquierda", "derecha") else ""}{vista}',
                         fontsize=13, fontweight='bold')
        for ax in ejes.flat:
            ax.axis('off')
        _titulo_rojo_negro(fig, titulo, subtitulo)
        _barra_color(fig, ejes, limites, 'Aumento de potencia al inicio de la crisis (dB)')
        if region_maxima:
            fig.text(0.45, 0.015, f'Máximo: {region_maxima}', ha='center', fontsize=13, color=ROJO_OSCURO,
                     fontweight='bold')
        fig.subplots_adjust(top=0.9, bottom=0.05, wspace=0.02, hspace=0.12, right=0.86)
    return fig


def figura_desde_arriba(origen, captura_arriba, limites, titulo, subtitulo):
    """Dos vistas desde arriba, lado a lado: lo que miden los electrodos y dónde lo sitúa sLORETA.

    Izquierda: mapa del cuero cabelludo (topomap). Cada electrodo con su aumento de potencia y el
        color interpolado entre ellos. Es lo que se mide directamente.
    Derecha: el cerebro 3D visto desde arriba, con el origen estimado (fuentes).
    Las dos con la misma regla: gris por debajo del 60 % del máximo y de amarillo a rojo oscuro por encima
    (cada una con su máximo, porque miden cosas distintas: electrodos y puntos del cerebro).
    """
    electrodos_db = origen['electrodos_db']
    limites_cabeza = limites_color(electrodos_db.values)
    with plt.rc_context(ESTILO_TIMES):
        fig, (ax_cabeza, ax_cerebro) = plt.subplots(1, 2, figsize=(10, 6))
        imagen, _ = mne.viz.plot_topomap(electrodos_db.values, origen['info'], axes=ax_cabeza, show=False,
                                         cmap=MAPA_COLOR, vlim=(limites_cabeza['fmin'], limites_cabeza['fmax']),
                                         contours=0, names=list(electrodos_db.index), sensors=True)
        _barra_color(fig, ax_cabeza, limites_cabeza, 'Aumento de potencia (dB)')
        ax_cabeza.set_title('Cuero cabelludo: lo que miden los electrodos', fontsize=13, fontweight='bold')

        ax_cerebro.imshow(captura_arriba)
        ax_cerebro.axis('off')
        ax_cerebro.set_title('Cerebro: origen estimado (sLORETA)', fontsize=13, fontweight='bold')
        _barra_color(fig, ax_cerebro, limites, 'Aumento de potencia (dB)')
        _titulo_rojo_negro(fig, titulo, subtitulo)
        fig.subplots_adjust(top=0.84, bottom=0.04, wspace=0.15)
    return fig
