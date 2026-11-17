"""
Regresión logística: suma ponderada de las características y una curva en S que la pasa a 0-1.

Motivo para cambiar: que cambie cómo se monta o se configura la regresión logística.

Imagen: fase_03/img/dos_modelos.png (regresión logística y XGBoost, explicados con un dibujo)

Es la más sencilla y la más fácil de explicar: cada característica tiene UN peso, con signo (positivo = empuja
hacia ⏳ preictal). En la línea base, antes se rellenan los NaN (canales sin señal) con la mediana de entrenamiento y
se pone cada característica en la misma escala (media 0, desviación 1), para que los pesos se puedan comparar.
C controla cuánto se frenan los pesos: C pequeño = modelo más prudente.
"""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NOMBRE = 'regresión logística'
REJILLA = {'C': [0.01, 0.1, 1.0]}
CLAVE_PESO = 'logisticregression__sample_weight'   # cómo se pasa el peso de cada ventana al .fit de la tubería


def crear(hiperparametros, semilla=42):
    """La regresión logística de la línea base (F03.02), sin entrenar: rellenar NaN -> misma escala -> logística.

    Parámetros:
        hiperparametros: una combinación de REJILLA.  semilla: fija el azar.
    """
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=3000, random_state=semilla, **hiperparametros))


def crear_para_metadatos():
    """La regresión logística de la auditoría de atajos (F02.05), sin entrenar: solo el modelo (la preparación de
    las columnas la pone el notebook)."""
    return LogisticRegression(max_iter=2000)


def pesos_explicativos(modelo):
    """El peso de cada característica, CON signo (ya están en la misma escala).

    Positivo = cuanto más alta, más ⏳ preictal; negativo = más 💤 interictal.
    """
    return modelo[-1].coef_[0]
