"""
Remuestreo de la señal a una frecuencia común.

Motivo para cambiar: que cambie la frecuencia de trabajo o cómo se remuestrea.
Trabaja con arrays de numpy (canales x muestras), sin MNE, para poder usarlo
igual en el TFM y en la demo en tiempo real.

TUSZ tiene ficheros a 250, 256, 400, 512 y 1000 Hz: todos se pasan a FS_OBJETIVO.
"""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

FS_OBJETIVO = 256   # Hz: la más frecuente en TUSZ y de sobra para filtrar hasta 40 Hz


def remuestrear(senal, fs_original, fs_objetivo=FS_OBJETIVO):
    """Cambia el número de muestras por segundo de una señal (p. ej. de 250 a 256 Hz).

    Qué hace:
        La onda es la misma; solo cambian los instantes en los que se mide. resample_poly sube y baja
        la frecuencia por factores enteros (250 -> 256 Hz es x128 / 125) y ya incluye el filtro
        antialiasing, así que no hace falta filtrar antes. Si la frecuencia ya es la objetivo,
        devuelve la señal sin tocarla.

    Parámetros:
        senal: matriz (canales x muestras) a fs_original.
        fs_original: frecuencia de muestreo de la señal (Hz).
        fs_objetivo: frecuencia a la que se quiere pasar (por defecto FS_OBJETIVO = 256 Hz).

    Devuelve:
        Matriz (canales x muestras nuevas) a fs_objetivo. Muestras nuevas = muestras x fs_objetivo / fs_original.

    Imagen: tfm_comun/imagenes/remuestreo_remuestrear.png
    """
    if round(fs_original) == round(fs_objetivo):
        return np.asarray(senal)
    fraccion = Fraction(round(fs_objetivo), round(fs_original))
    return resample_poly(senal, fraccion.numerator, fraccion.denominator, axis=-1)
