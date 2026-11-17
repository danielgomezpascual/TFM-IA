"""
Modelos clásicos (sin red neuronal): cada modelo en su propio fichero (un solo motivo para cambiar: ese modelo).

    regresion_logistica.py -> regresión logística: la de la línea base (F03.02) y la de la auditoría de atajos (F02.05).
    xgboost_arboles.py     -> XGBoost: árboles de decisión encadenados (línea base, F03.02).
    arboles_histograma.py  -> HistGradientBoosting de scikit-learn: árboles de la auditoría de atajos (F02.05).

Cada fichero de la línea base lleva su nombre para las tablas (NOMBRE), su rejilla de hiperparámetros (REJILLA),
cómo se crea sin entrenar (crear) y cuánto usa cada característica (pesos_explicativos).
Cómo se eligen, entrenan y puntúan: prediccion/linea_base.py.
"""
