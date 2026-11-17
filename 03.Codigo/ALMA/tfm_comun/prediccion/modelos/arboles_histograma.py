"""
Árboles por histogramas (HistGradientBoosting de scikit-learn): árboles encadenados, como XGBoost, que captan
combinaciones y umbrales.

Motivo para cambiar: que cambie cómo se configuran los árboles de la auditoría de atajos.

Se usan en la auditoría de atajos (F02.05): ¿se puede adivinar la etiqueta solo con metadatos (canales a 0, montaje,
minuto del fichero...)? Si estos árboles lo consiguen, hay una pista tramposa que la red podría aprovechar.
"""

from sklearn.ensemble import HistGradientBoostingClassifier

NOMBRE = 'árboles'


def crear(semilla=42, iteraciones=200):
    """HistGradientBoosting sin entrenar (200 árboles).  semilla: fija el azar."""
    return HistGradientBoostingClassifier(max_iter=iteraciones, random_state=semilla)
