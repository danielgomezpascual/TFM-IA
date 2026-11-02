"""
Métricas: cómo se mide lo bien que separa un modelo (o una variable) las ⏳ preictales de las 💤 interictales.

Motivo para cambiar: que cambie una métrica o cómo se calcula su intervalo de confianza.
La usan fase_02/05_auditoria_atajos y la Fase 3 (aquí irán también la sensibilidad por crisis y las FA/h).

Por qué un intervalo POR PACIENTE
=================================
Las ventanas de un mismo paciente se parecen mucho entre sí: 10.000 ventanas de 30 pacientes no son
10.000 datos independientes, son más bien 30. Si el intervalo se calcula sorteando ventanas, sale
estrechísimo y engaña. Se sortean PACIENTES (bootstrap por paciente): cada repetición elige pacientes
al azar, con repetición, y recalcula la AUROC con todas sus ventanas. El intervalo dice cuánto cambiaría
la cifra con otros pacientes parecidos, que es lo que importa.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 500    # repeticiones del bootstrap: 500 basta para un intervalo del 95 % estable a 2 decimales
NIVEL = 0.95         # intervalo de confianza del 95 %


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
    etiqueta, puntuacion = np.asarray(etiqueta), np.asarray(puntuacion, dtype=float)
    codigos, unicos = pd.factorize(np.asarray(pacientes))
    auroc = float(roc_auc_score(etiqueta, puntuacion))
    rng = np.random.default_rng(semilla)
    muestras = []
    while len(muestras) < n_bootstrap:
        veces = np.bincount(rng.integers(0, len(unicos), len(unicos)), minlength=len(unicos))
        peso = veces[codigos]
        if np.unique(etiqueta[peso > 0]).size < 2:
            continue
        muestras.append(roc_auc_score(etiqueta, puntuacion, sample_weight=peso))
    inferior, superior = np.percentile(muestras, [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2])
    return auroc, float(inferior), float(superior)
