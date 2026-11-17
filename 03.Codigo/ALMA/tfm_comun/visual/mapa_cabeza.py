"""
Mapa de color sobre la cabeza (topomapa): un valor por electrodo, rellenando la cabeza entre ellos.

Motivo para cambiar: que cambie cómo se interpola o se colorea el mapa de la cabeza.
La silueta y las posiciones de los electrodos son las de cabeza.py (vista desde arriba, nariz arriba),
para que todas las cabezas del TFM se vean igual.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator

from tfm_comun.visual.cabeza import POSICIONES_1020, dibujar_contorno

RESOLUCION = 120   # puntos por lado de la rejilla que se colorea


def dibujar_mapa_cabeza(ax, valores, cmap, vmin=None, vmax=None, marcar_maximo=True):
    """Colorea la cabeza a partir de un valor por electrodo.

    Qué hace:
        1. Coloca cada valor en su electrodo (POSICIONES_1020 de cabeza.py).
        2. Rellena el resto de la cabeza interpolando con una superficie suave (RBF "thin plate spline",
           como una lámina flexible que pasa por todos los electrodos) y recorta fuera del círculo.
        3. Dibuja la silueta (nariz y orejas), los electrodos como puntos y, si se pide, el nombre
           del electrodo con el valor más alto.

    Parámetros:
        ax: eje de matplotlib.
        valores: {electrodo: valor}. Los electrodos sin posición en el esquema se ignoran.
        cmap: escala de color de matplotlib. vmin, vmax: límites del color (None = mínimo y máximo).
        marcar_maximo: escribir el electrodo con el valor más alto.

    Devuelve:
        La imagen de matplotlib (para una barra de color, si hace falta).
    """
    valores = {e: v for e, v in valores.items() if e in POSICIONES_1020 and np.isfinite(v)}
    posiciones = np.array([POSICIONES_1020[e] for e in valores])
    datos = np.array(list(valores.values()))

    eje = np.linspace(-1, 1, RESOLUCION)
    x, y = np.meshgrid(eje, eje)
    rejilla = RBFInterpolator(posiciones, datos, kernel='thin_plate_spline', smoothing=1e-3)(
        np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)
    rejilla[x ** 2 + y ** 2 > 1] = np.nan

    imagen = ax.imshow(rejilla, extent=(-1, 1, -1, 1), origin='lower', cmap=cmap,
                       vmin=datos.min() if vmin is None else vmin, vmax=datos.max() if vmax is None else vmax)
    dibujar_contorno(ax)
    ax.scatter(posiciones[:, 0], posiciones[:, 1], s=6, color='black', zorder=3)
    if marcar_maximo:
        electrodo = max(valores, key=valores.get)
        px, py = POSICIONES_1020[electrodo]
        ax.text(px, py + 0.09, electrodo, ha='center', va='bottom', fontsize=8, fontweight='bold', zorder=4)
    return imagen
