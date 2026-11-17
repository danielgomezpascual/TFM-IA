"""Sensibilidad con especificidad fija: un punto de la curva ROC que se entiende en palabras.

Motivo para cambiar: que cambie cómo se elige el punto de la curva que se da.
"""

import numpy as np


def sensibilidad_con_especificidad(etiqueta, puntuacion, especificidad=0.9):
    """Sensibilidad cuando el umbral se pone para acertar `especificidad` de las 💤 interictales.

    Qué hace:
        1. Umbral = el valor que deja por debajo el 90 % (por defecto) de las puntuaciones 💤.
        2. Sensibilidad = qué parte de las ⏳ preictales pasan de ese umbral.
    Es una forma de dar UN punto de la curva ROC que se entiende: "si solo se permite equivocarse en el 10 %
    de las ventanas tranquilas, ¿cuántas de las de antes de una crisis se detectan?".

    Devuelve:
        (sensibilidad, umbral).
    """
    etiqueta, puntuacion = np.asarray(etiqueta), np.asarray(puntuacion, dtype=float)
    umbral = float(np.quantile(puntuacion[etiqueta == 0], especificidad))
    return float((puntuacion[etiqueta == 1] > umbral).mean()), umbral
