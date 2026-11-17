"""
Panel de frecuencias de un trozo de EEG: el espectro y una cabeza por banda.

Motivo para cambiar: que cambie cómo se enseñan las frecuencias de la ventana del visor.
Los cálculos están en senal/espectro.py; las cabezas, en visual/mapa_cabeza.py.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tfm_comun.senal.bandas import BANDAS_HZ
from tfm_comun.senal.espectro import RANGO_HZ, frecuencia_dominante, potencia_relativa_bandas
from tfm_comun.visual.colores import COLORES_BANDA, COLORES_ESQUEMA
from tfm_comun.visual.mapa_cabeza import dibujar_mapa_cabeza
from tfm_comun.visual.tipografia import ESTILO_TIMES


def figura_frecuencias(frecuencias, potencia, electrodos, titulo=''):
    """Espectro medio de la ventana (izquierda) y dónde está cada banda en la cabeza (derecha).

    Qué enseña:
        - Espectro: potencia de cada frecuencia (media de los electrodos), con las bandas coloreadas
          y el ritmo dominante marcado en rojo (frecuencia_dominante() de espectro.py). Eje vertical en escala logarítmica, porque las
          frecuencias lentas tienen muchísima más potencia que las rápidas.
        - Una cabeza por banda: en qué electrodos pesa más esa banda (% de la potencia del electrodo).
          Cada cabeza tiene su propia escala (de blanco al color de la banda): el color dice DÓNDE,
          y el % del título dice CUÁNTO pesa la banda de media.

    Parámetros:
        frecuencias, potencia: de espectro() (potencia: electrodos x frecuencias, en µV²/Hz).
        electrodos: nombre de cada fila de potencia ('FP1', 'C3'...).
        titulo: texto de arriba.

    Devuelve:
        La figura de matplotlib.
    """
    media = potencia.mean(axis=0)
    pico_hz, banda_pico = frecuencia_dominante(frecuencias, media)
    relativa = potencia_relativa_bandas(frecuencias, potencia)

    with plt.rc_context(ESTILO_TIMES):
        fig = plt.figure(figsize=(15, 4.2))
        rejilla = fig.add_gridspec(1, 1 + len(BANDAS_HZ), width_ratios=[2.4] + [1] * len(BANDAS_HZ), wspace=0.05)

        ax = fig.add_subplot(rejilla[0, 0])
        for banda, (desde, hasta) in BANDAS_HZ.items():
            ax.axvspan(desde, hasta, color=COLORES_BANDA[banda], alpha=0.12, lw=0)
            ax.text((desde * hasta) ** 0.5, 1.01, banda, transform=ax.get_xaxis_transform(), ha='center',
                    va='bottom', fontsize=9, color=COLORES_BANDA[banda], fontweight='bold')
        ax.semilogy(frecuencias, media, color=COLORES_ESQUEMA['negro'], lw=1.2)
        ax.axvline(pico_hz, color=COLORES_ESQUEMA['rojo'], lw=1, ls='--')
        ax.text(pico_hz, 0.93, f' ritmo dominante {pico_hz:.1f} Hz ({banda_pico})', transform=ax.get_xaxis_transform(),
                color=COLORES_ESQUEMA['rojo'], fontsize=10, fontweight='bold', va='top')
        ax.set_xscale('log')
        ticks = [0.5, 1, 2, 4, 8, 13, 20, 30, 45]
        ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
        ax.set_xlim(*RANGO_HZ)
        ax.set_xlabel('frecuencia (Hz)')
        ax.set_ylabel('potencia (µV²/Hz)')
        ax.set_title(titulo, loc='left', fontsize=11, pad=16)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)

        for k, banda in enumerate(BANDAS_HZ):
            ax_cabeza = fig.add_subplot(rejilla[0, k + 1])
            cmap = LinearSegmentedColormap.from_list(banda, ['white', COLORES_BANDA[banda]])
            dibujar_mapa_cabeza(ax_cabeza, dict(zip(electrodos, relativa[banda])), cmap)
            ax_cabeza.set_title(f'{banda} · {100 * np.mean(relativa[banda]):.0f} %', fontsize=11,
                                color=COLORES_BANDA[banda], fontweight='bold')
        fig.subplots_adjust(left=0.05, right=0.995, top=0.86, bottom=0.13)
    return fig
