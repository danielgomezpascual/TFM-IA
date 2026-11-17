"""Rango de frecuencias de la entrada: qué frecuencias ve la red y los suelos numéricos comunes a todas las entradas.

Motivo para cambiar: que cambie el rango de frecuencias o la resolución (trozos de 1 s = 1 Hz).

Por qué 1-40 Hz:
    - 0 Hz (el nivel medio de la señal) no dice nada del ritmo cerebral;
    - la señal de F02.02 viene filtrada a 0,5-40 Hz: por encima de 40 Hz solo queda ruido del filtro.
Por qué trozos de 1 s: con 1 s de señal, la FFT separa las frecuencias de 1 en 1 Hz (resolución = 1 / duración).
"""

import numpy as np


SEGMENTO_S = 1.0              # trozos de 1 s para Welch (resolución de 1 Hz), como en F03.02


FRECUENCIA_MIN_HZ = 1         # 0 Hz (el nivel medio) no dice nada del ritmo cerebral


FRECUENCIA_MAX_HZ = 40        # la señal viene filtrada a 0,5-40 Hz


POTENCIA_MINIMA = 1e-6        # suelo antes del logaritmo (log10(0) no existe)


ENERGIA_MINIMA = 1e-12        # por debajo de esto, el canal se considera sin señal (relleno con ceros)


def frecuencias_entrada(fs=256, minima=FRECUENCIA_MIN_HZ, maxima=FRECUENCIA_MAX_HZ):
    """Las frecuencias (Hz) que entran en la red: [1, 2, ..., 40] con fs = 256."""
    paso = 1 / SEGMENTO_S
    return np.arange(minima, maxima + paso / 2, paso)
