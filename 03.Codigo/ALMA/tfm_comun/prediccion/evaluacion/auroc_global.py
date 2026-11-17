"""AUROC global: la nota con todas las ventanas juntas (de todos los pacientes) y su intervalo por paciente.

Motivo para cambiar: que cambie cómo se calcula la AUROC global o su intervalo.

Imagen: fase_03/img/auc_roc.png (qué es la AUC-ROC y por qué se mide también por paciente)

Qué es la AUROC (o AUC-ROC)
===========================
Se coge al azar una ventana ⏳ preictal y una 💤 interictal: la AUROC es la probabilidad de que el modelo dé
más puntuación a la ⏳. 0,5 = azar (tirar una moneda); 1 = perfecto. No depende de cuántas ventanas haya de
cada clase, así que se puede comparar entre grupos con desbalances distintos (validación, evaluación, controles).
El intervalo se calcula sorteando pacientes (red de seguridad explicada en prediccion/evaluacion/bootstrap.py).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tfm_comun.prediccion.evaluacion.bootstrap import N_BOOTSTRAP, NIVEL


def muestras_auroc_bootstrap(etiqueta, puntuacion, pacientes, n_bootstrap=N_BOOTSTRAP, semilla=42):
    """Las n_bootstrap AUROC del bootstrap por paciente (el paso 2 de auroc_con_intervalo()).

    Sirve para dibujar de dónde sale el intervalo: un histograma de estas AUROC con sus percentiles 2,5 y 97,5
    marcados es exactamente el intervalo que da auroc_con_intervalo() con la misma semilla.

    Devuelve:
        Array con n_bootstrap AUROC (una por sorteo de pacientes).
    """
    etiqueta, puntuacion = np.asarray(etiqueta), np.asarray(puntuacion, dtype=float)
    codigos, unicos = pd.factorize(np.asarray(pacientes))
    rng = np.random.default_rng(semilla)
    muestras = []
    while len(muestras) < n_bootstrap:
        veces = np.bincount(rng.integers(0, len(unicos), len(unicos)), minlength=len(unicos))
        peso = veces[codigos]
        if np.unique(etiqueta[peso > 0]).size < 2:
            continue
        muestras.append(roc_auc_score(etiqueta, puntuacion, sample_weight=peso))
    return np.array(muestras)


def auroc_con_intervalo(etiqueta, puntuacion, pacientes, n_bootstrap=N_BOOTSTRAP, nivel=NIVEL, semilla=42):
    """AUROC y su intervalo de confianza con bootstrap por paciente.

    Qué hace:
        1. AUROC con todas las ventanas.
        2. n_bootstrap veces: sortea tantos pacientes como hay, con repetición, y recalcula la AUROC dando a
           cada ventana un peso = veces que ha salido su paciente (es lo mismo que copiar sus ventanas, pero
           sin copiar nada). Si en un sorteo falta una de las dos clases, se repite el sorteo.
        3. El intervalo son los percentiles (1 - nivel) / 2 y (1 + nivel) / 2 de esas AUROC (2,5 y 97,5 %).

    Parámetros:
        etiqueta: 1 = ⏳ preictal, 0 = 💤 interictal.  puntuacion: lo que da el modelo (más alto = más preictal).
        pacientes: el paciente de cada ventana.  n_bootstrap, nivel, semilla: ver arriba.

    Devuelve:
        (auroc, inferior, superior).
    Ejemplo: (0,78, 0,70, 0,85) -> la AUROC es 0,78 y con otros pacientes parecidos estaría entre 0,70 y 0,85.
    """
    auroc = float(roc_auc_score(np.asarray(etiqueta), np.asarray(puntuacion, dtype=float)))
    muestras = muestras_auroc_bootstrap(etiqueta, puntuacion, pacientes, n_bootstrap, semilla)
    inferior, superior = np.percentile(muestras, [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2])
    return auroc, float(inferior), float(superior)


def diferencia_auroc_con_intervalo(etiqueta, puntuacion_a, puntuacion_b, pacientes, n_bootstrap=N_BOOTSTRAP, nivel=NIVEL, semilla=42):
    """Diferencia de AUROC entre dos modelos medidos con las MISMAS ventanas (A - B) y su intervalo por paciente.

    Qué hace:
        Igual que auroc_con_intervalo(), pero en cada sorteo de pacientes calcula las dos AUROC con los mismos
        pacientes y guarda la resta. Así la comparación es "emparejada": la suerte de qué pacientes salen afecta
        igual a los dos modelos, y el intervalo de la diferencia es mucho más estrecho que comparar dos intervalos.

    Devuelve:
        (diferencia, inferior, superior). Si el intervalo no incluye el 0, la diferencia es real (con un 95 % de confianza).
    Ejemplo: (0,04, 0,01, 0,07) -> A es 0,04 mejor que B y, con otros pacientes parecidos, entre 0,01 y 0,07 mejor.
    """
    etiqueta = np.asarray(etiqueta)
    a, b = np.asarray(puntuacion_a, dtype=float), np.asarray(puntuacion_b, dtype=float)
    codigos, unicos = pd.factorize(np.asarray(pacientes))
    diferencia = float(roc_auc_score(etiqueta, a) - roc_auc_score(etiqueta, b))
    rng = np.random.default_rng(semilla)
    muestras = []
    while len(muestras) < n_bootstrap:
        peso = np.bincount(rng.integers(0, len(unicos), len(unicos)), minlength=len(unicos))[codigos]
        if np.unique(etiqueta[peso > 0]).size < 2:
            continue
        muestras.append(roc_auc_score(etiqueta, a, sample_weight=peso) - roc_auc_score(etiqueta, b, sample_weight=peso))
    inferior, superior = np.percentile(muestras, [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2])
    return diferencia, float(inferior), float(superior)
