"""
Dibujo de la cabeza vista desde arriba con los electrodos del sistema 10-20.

Motivo para cambiar: que cambie cómo se dibujan la cabeza, los electrodos o el montaje.
Las posiciones son un esquema 2D (nariz arriba, oreja izquierda a la izquierda),
no medidas reales de un paciente.

Imagen: tfm_comun/imagenes/cabeza_capas.png (contorno, electrodos y pares, capa a capa)
"""

import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, FancyArrowPatch

from tfm_comun.dataset.canales import region_electrodo
from tfm_comun.visual.colores import COLORES_REGION

RADIO_ANILLO = 0.8   # anillo exterior: FP1, F7, T3, T5, O1... (10 % por encima de nasion e inion)


def _en_anillo(grados_desde_delante, lado):
    """Coordenadas (x, y) de un electrodo del anillo exterior de la cabeza.

    El ángulo se mide desde delante (0° = nariz, 90° = oreja, 180° = nuca) y lado dice hacia dónde
    se gira (-1 = izquierda, +1 = derecha). Se usa para colocar FP1, F7, T3, T5, O1... en pasos de 36°,
    que es de donde viene el nombre "10-20" (distancias del 10 % y 20 %).
    """
    angulo = np.deg2rad(90 - lado * grados_desde_delante)
    return RADIO_ANILLO * np.cos(angulo), RADIO_ANILLO * np.sin(angulo)


POSICIONES_1020 = {
    # Anillo exterior, en pasos de 36° (los "10-20": 10 % y 20 % de la distancia)
    'FP1': _en_anillo(18, -1), 'FP2': _en_anillo(18, 1),
    'F7': _en_anillo(54, -1), 'F8': _en_anillo(54, 1),
    'T3': _en_anillo(90, -1), 'T4': _en_anillo(90, 1),
    'T5': _en_anillo(126, -1), 'T6': _en_anillo(126, 1),
    'O1': _en_anillo(162, -1), 'O2': _en_anillo(162, 1), 'OZ': (0, -RADIO_ANILLO),
    # Interior
    'F3': (-0.34, 0.44), 'FZ': (0, 0.4), 'F4': (0.34, 0.44),
    'C3': (-0.4, 0), 'CZ': (0, 0), 'C4': (0.4, 0),
    'P3': (-0.34, -0.44), 'PZ': (0, -0.4), 'P4': (0.34, -0.44),
    # Fuera del anillo: temporales anteriores y orejas
    'T1': (-0.93, 0.3), 'T2': (0.93, 0.3),
    'A1': (-1.13, 0), 'A2': (1.13, 0),
}


def dibujar_contorno(ax):
    """Dibuja la silueta de la cabeza vista desde arriba: círculo, nariz (arriba) y orejas.

    También fija los límites del eje, la misma escala en x e y (para que el círculo no salga ovalado)
    y quita los ejes. Es la primera capa: después se añaden electrodos y pares.

    Imagen: tfm_comun/imagenes/cabeza_capas.png
    """
    ax.add_patch(Circle((0, 0), 1, fill=False, linewidth=2, color='dimgray'))
    ax.add_patch(Polygon([(-0.12, 0.99), (0, 1.15), (0.12, 0.99)], closed=False, fill=False,
                         linewidth=2, color='dimgray'))
    for lado in (-1, 1):
        ax.add_patch(Ellipse((lado * 1.06, 0), 0.14, 0.4, fill=False, linewidth=2, color='dimgray'))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.15, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')


def dibujar_electrodos(ax, electrodos, tamano=0.085, etiquetas=True, fuente=8, color=None):
    """Dibuja electrodos como círculos con su nombre, coloreados por región de la cabeza.

    Parámetros:
        ax: eje de matplotlib (normalmente después de dibujar_contorno).
        electrodos: nombres de los electrodos a dibujar ('FP1', 'C3'...). Los que no tienen posición
            en el esquema (PG1, SP1...) se ignoran sin error.
        tamano: radio del círculo. etiquetas: si se escribe el nombre dentro. fuente: tamaño de letra.
        color: un color para todos; None = el de su región (COLORES_REGION).

    Imagen: tfm_comun/imagenes/cabeza_capas.png
    """
    for nombre in electrodos:
        if nombre not in POSICIONES_1020:
            continue
        x, y = POSICIONES_1020[nombre]
        relleno = color or COLORES_REGION[region_electrodo(nombre)]
        ax.add_patch(Circle((x, y), tamano, facecolor=relleno, edgecolor='black', linewidth=0.8, zorder=3))
        if etiquetas:
            ax.text(x, y, nombre, ha='center', va='center', fontsize=fuente, zorder=4,
                    fontweight='bold', color='white' if relleno not in ('gold',) else 'black')


def dibujar_pares(ax, pares, color='black', ancho=1.6):
    """Dibuja una flecha ánodo -> cátodo por cada par bipolar (cada flecha es un canal = ánodo - cátodo).

    Sirve para enseñar un montaje, p. ej. las cadenas TCP: dibujar_pares(ax, CADENAS_TCP['central']).
    Los pares con algún electrodo sin posición en el esquema se saltan.

    Imagen: tfm_comun/imagenes/cabeza_capas.png
    """
    for anodo, catodo in pares:
        if anodo in POSICIONES_1020 and catodo in POSICIONES_1020:
            ax.add_patch(FancyArrowPatch(POSICIONES_1020[anodo], POSICIONES_1020[catodo],
                                         arrowstyle='-|>', mutation_scale=14, color=color,
                                         linewidth=ancho, shrinkA=13, shrinkB=13, zorder=2))
