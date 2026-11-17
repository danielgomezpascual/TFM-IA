"""
Dibujo de un trozo de EEG como en un electroencefalógrafo: un canal debajo de otro, con las crisis sombreadas.

Motivo para cambiar: que cambie cómo se ve el trazado del EEG.
Solo dibuja: no lee ficheros ni filtra (eso es de dataset/ y senal/). Lo usa el visor (visual/visor_eeg.py).
"""

import matplotlib.pyplot as plt
import numpy as np

from tfm_comun.senal.montaje import CADENAS_TCP
from tfm_comun.visual.colores import COLORES_ESQUEMA, COLORES_ETIQUETA
from tfm_comun.visual.tipografia import ESTILO_TIMES


def dibujar_trazado(senal_uv, canales, fs, inicio_s, crisis, escala_uv=100, titulo=''):
    """Dibuja un trozo de EEG (canales apilados) con las crisis que caen dentro sombreadas.

    Qué hace:
        1. Pone cada canal en su "carril": el canal k se dibuja desplazado k * escala_uv hacia abajo.
           Así, una onda de escala_uv µV ocupa justo la separación entre dos canales.
        2. Separa con una línea roja fina las 5 cadenas del montaje TCP (temporal izquierda, derecha...).
        3. Sombrea cada crisis con el color de su tipo (colores.py) y escribe el tipo arriba.
        4. Dibuja una barra de escala abajo a la derecha: escala_uv µV en vertical y 1 s en horizontal.

    Parámetros:
        senal_uv: matriz (canales x muestras) en µV, solo el trozo a dibujar.
        canales: nombre de cada fila ('FP1-F7'...).
        fs: frecuencia de muestreo (Hz).
        inicio_s: segundo del fichero en el que empieza el trozo (para el eje de tiempo).
        crisis: tabla de crisis_del_fichero() (inicio_s, fin_s, tipo); se dibujan las que se solapan.
        escala_uv: µV entre carriles. Más pequeño = ondas más grandes.
        titulo: texto de arriba.

    Devuelve:
        La figura de matplotlib.
    """
    n_canales, n_muestras = senal_uv.shape
    tiempo = inicio_s + np.arange(n_muestras) / fs
    fin_s = tiempo[-1]

    with plt.rc_context(ESTILO_TIMES):
        fig, ax = plt.subplots(figsize=(15, 0.38 * n_canales + 1.6))
        for k in range(n_canales):
            ax.plot(tiempo, senal_uv[k] - k * escala_uv, color=COLORES_ESQUEMA['negro'], lw=0.6)

        for k in _fronteras_cadenas(canales):
            ax.axhline(-(k - 0.5) * escala_uv, color=COLORES_ESQUEMA['rojo'], lw=0.5, alpha=0.5)

        for _, c in crisis[(crisis.fin_s > inicio_s) & (crisis.inicio_s < fin_s)].iterrows():
            color = COLORES_ETIQUETA.get(c.tipo, 'black')
            ax.axvspan(max(c.inicio_s, inicio_s), min(c.fin_s, fin_s), color=color, alpha=0.15, lw=0)
            ax.text(max(c.inicio_s, inicio_s), 0.6 * escala_uv, f' crisis {c.tipo}', color=color,
                    fontsize=11, fontweight='bold', va='bottom')

        _barra_escala(ax, fin_s, n_canales, escala_uv)
        ax.set_yticks(-np.arange(n_canales) * escala_uv, canales, fontsize=9)
        ax.set_ylim(-(n_canales - 0.3) * escala_uv - escala_uv, 1.5 * escala_uv)
        ax.set_xlim(tiempo[0], fin_s)
        ax.set_xlabel('tiempo desde el inicio del fichero (s)')
        ax.set_title(titulo, loc='left', fontsize=12)
        ax.grid(axis='x', color=COLORES_ESQUEMA['gris'], lw=0.3, alpha=0.5)
        for lado in ('top', 'right', 'left'):
            ax.spines[lado].set_visible(False)
        fig.tight_layout()
    return fig


def _fronteras_cadenas(canales):
    """Índices de los canales con los que empieza una cadena TCP nueva (menos la primera)."""
    primeros = {f'{a}-{c}' for pares in CADENAS_TCP.values() for a, c in pares[:1]}
    return [k for k, canal in enumerate(canales) if canal in primeros and k > 0]


def _barra_escala(ax, fin_s, n_canales, escala_uv):
    """Barra en L abajo a la derecha: escala_uv µV de alto y 1 s de ancho, para leer amplitudes y tiempos."""
    x, y = fin_s - 1.3, -(n_canales - 0.2) * escala_uv - 0.9 * escala_uv
    ax.plot([x, x, x + 1], [y + escala_uv, y, y], color=COLORES_ESQUEMA['rojo'], lw=1.5, clip_on=False)
    ax.text(x - 0.08, y + escala_uv / 2, f'{escala_uv} µV', ha='right', va='center', fontsize=9,
            color=COLORES_ESQUEMA['rojo'])
    ax.text(x + 0.5, y - 0.08 * escala_uv, '1 s', ha='center', va='top', fontsize=9, color=COLORES_ESQUEMA['rojo'])
