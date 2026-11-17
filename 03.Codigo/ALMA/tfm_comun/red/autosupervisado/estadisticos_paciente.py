"""Estadísticos por paciente (F03.07): la mediana y la escala de cada número para cada paciente del corpus.

Motivo para cambiar: que cambie cómo se resume "lo normal" de un paciente.

La misma idea que en la Fase 3 (cada paciente frente a sí mismo, prediccion/normalizacion_paciente.py): a cada número
se le resta la mediana de SU paciente y se divide por su escala (rango intercuartílico / 1,349, que en una campana
de Gauss es la desviación típica). Se usa una muestra de ventanas por paciente: con 400 basta y es mucho más rápido.
"""

import warnings

import numpy as np

from tfm_comun.prediccion.normalizacion_paciente import EPSILON, ESCALA_IQR


def estadisticos_por_paciente(valores, paciente, por_paciente=400, semilla=42):
    """Mediana y escala de cada número para cada paciente, con una muestra de sus ventanas.

    Devuelve:
        (indice_paciente, medianas, escalas): indice_paciente = int por ventana; medianas, escalas = float32
        (pacientes, números).
    """
    rng = np.random.default_rng(semilla)
    nombres, indice = np.unique(paciente, return_inverse=True)
    medianas = np.zeros((len(nombres), valores.shape[1]), dtype=np.float32)
    escalas = np.ones_like(medianas)
    for i in range(len(nombres)):
        posiciones = np.where(indice == i)[0]
        if len(posiciones) > por_paciente:
            posiciones = np.sort(rng.choice(posiciones, por_paciente, replace=False))
        bloque = valores[posiciones].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mediana = np.nanmedian(bloque, axis=0)
            q1, q3 = np.nanpercentile(bloque, [25, 75], axis=0)
        medianas[i] = np.nan_to_num(mediana)
        escalas[i] = np.nan_to_num((q3 - q1) / ESCALA_IQR, nan=1.0) + EPSILON
    return indice.astype(np.int64), medianas, escalas
