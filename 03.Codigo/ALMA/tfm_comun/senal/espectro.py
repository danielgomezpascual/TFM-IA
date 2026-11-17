"""
Espectro de un trozo de EEG: cuánta potencia hay en cada frecuencia y en cada banda.

Motivo para cambiar: que cambie cómo se calcula el espectro (método, resolución o rango).
Las bandas (delta, theta...) están en bandas.py. Lo usa el visor de EEG para el panel de frecuencias.
"""

import numpy as np
from scipy.signal import welch

from tfm_comun.senal.bandas import BANDAS_HZ

RANGO_HZ = (0.5, 45)   # de la banda delta a la gamma (lo que está fuera es deriva, red eléctrica o músculo)
SEGMENTO_S = 2         # trozos de Welch: 2 s = resolución de 0,5 Hz


def espectro(senal, fs):
    """Espectro de potencia de cada canal con el método de Welch.

    Qué hace:
        Welch parte la señal en trozos de SEGMENTO_S que se solapan, calcula el espectro de cada uno
        y los promedia: sale mucho menos ruidoso que el espectro de la señal entera de golpe.
        Se queda solo con RANGO_HZ.

    Parámetros:
        senal: matriz (canales x muestras), en µV.
        fs: frecuencia de muestreo (Hz).

    Devuelve:
        (frecuencias, potencia): frecuencias en Hz y potencia (canales x frecuencias) en µV²/Hz.
    """
    muestras = min(int(SEGMENTO_S * fs), senal.shape[-1])
    frecuencias, potencia = welch(senal, fs=fs, nperseg=muestras, axis=-1)
    dentro = (frecuencias >= RANGO_HZ[0]) & (frecuencias <= RANGO_HZ[1])
    return frecuencias[dentro], potencia[..., dentro]


def potencia_relativa_bandas(frecuencias, potencia):
    """Parte de la potencia de cada canal que cae en cada banda (de 0 a 1; las bandas suman 1).

    Devuelve:
        {banda: array con un valor por canal}. Ejemplo: relativa['delta'][3] = 0.6 -> el 60 % de la
        potencia del canal 3 está en delta.
    """
    total = potencia.sum(axis=-1)
    return {banda: potencia[..., (frecuencias >= desde) & (frecuencias < hasta)].sum(axis=-1) / total
            for banda, (desde, hasta) in BANDAS_HZ.items()}


def frecuencia_dominante(frecuencias, potencia_media):
    """Ritmo que más destaca en la ventana (Hz), entre 1 y 30 Hz, y la banda en la que cae.

    El EEG siempre tiene más potencia cuanto más lenta es la frecuencia (cae como 1/f), así que el
    máximo "tal cual" saldría casi siempre en la frecuencia más baja. Para ver qué ritmo DESTACA se
    multiplica la potencia por la frecuencia (quita esa caída) y se busca el máximo de lo que queda.
    """
    dentro = (frecuencias >= 1) & (frecuencias <= 30)
    pico = float(frecuencias[dentro][np.argmax((potencia_media * frecuencias)[dentro])])
    banda = next((b for b, (desde, hasta) in BANDAS_HZ.items() if desde <= pico < hasta), None)
    return pico, banda
