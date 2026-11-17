"""Resumen del contexto (para XGBoost, F03.03): de K ventanas x 110 números a 330 números por fila.

Motivo para cambiar: que cambie qué se resume del contexto.

Para cada una de las 110 características (prediccion/caracteristicas.py):
    - media: el valor típico en los últimos C segundos;
    - variación: su desviación típica (¿se ha movido mucho?);
    - cambio: la ventana actual menos la media (¿lo de ahora es distinto de lo de antes?).
Con contexto de 4 s (K = 1) la variación y el cambio son 0: es exactamente la línea base de F03.02.
"""

import warnings

import numpy as np


RESUMENES = ['media', 'variación', 'cambio']


def resumir_contexto(cubo):
    """De (filas, k, 110) a (filas, 330): media, variación y cambio de cada característica en el contexto.

    Los NaN (canales sin señal) se ignoran al promediar; si un canal no tiene señal en todo el contexto, queda NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)   # medias de solo NaN: quedan NaN, a propósito
        media = np.nanmean(cubo, axis=1)
        variacion = np.nanstd(cubo, axis=1)
    cambio = cubo[:, -1, :] - media
    return np.concatenate([media, variacion, cambio], axis=1).astype(np.float32)


def nombres_contexto(nombres):
    """Nombres de las 330 columnas de resumir_contexto(): 'media · FP1-F7 · delta'..."""
    return [f'{resumen} · {n}' for resumen in RESUMENES for n in nombres]
