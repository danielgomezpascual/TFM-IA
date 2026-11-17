"""
XGBoost: cientos de árboles de decisión pequeños; cada uno corrige los errores de los anteriores.

Motivo para cambiar: que cambie cómo se configura XGBoost.

Imagen: fase_03/img/dos_modelos.png (regresión logística y XGBoost, explicados con un dibujo)

Capta combinaciones ("mucha delta en T3-T5 Y poca alfa en O1") que la logística no ve. Es el que mejor funciona
entre los clásicos en TUSZ (Roy 2020: 0,561 de F1 por paciente, mejor que sus CNN). Admite NaN sin rellenarlos.
max_depth = cuántas preguntas encadena cada árbol.
"""

import numpy as np
from xgboost import XGBClassifier

NOMBRE = 'XGBoost'
REJILLA = {'max_depth': [3, 5], 'n_estimators': [300], 'learning_rate': [0.05]}
CLAVE_PESO = 'sample_weight'


def crear(hiperparametros, semilla=42):
    """XGBoost sin entrenar.

    Parámetros:
        hiperparametros: una combinación de REJILLA.
        semilla: fija el azar (XGBoost sortea ventanas y características en cada árbol).
    """
    return XGBClassifier(subsample=0.8, colsample_bytree=0.8, tree_method='hist', eval_metric='auc',
                         random_state=semilla, n_jobs=-1, **hiperparametros)


def pesos_explicativos(modelo):
    """La "ganancia" de cada característica (cuánto mejoran los árboles al usarla), sin signo, repartida para que
    sume 1 (mismo orden que las características)."""
    ganancia = modelo.get_booster().get_score(importance_type='gain')
    valores = np.array([ganancia.get(f'f{i}', 0.0) for i in range(modelo.n_features_in_)])
    return valores / valores.sum() if valores.sum() > 0 else valores
