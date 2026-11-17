"""
Línea base: los modelos clásicos (sin red neuronal) con los que se compara la red de la Fase 3.

Motivo para cambiar: que cambien los modelos clásicos, sus hiperparámetros o cómo se entrenan.
Reciben las características de caracteristicas.py (110 números por ventana) y devuelven una puntuación
entre 0 y 1: lo probable que es que la ventana sea ⏳ preictal.

Los dos modelos (los que recomienda la síntesis del estado del arte para empezar)
==================================================================================
Cada uno en su fichero de prediccion/modelos/ (cómo se crea, su rejilla y cómo se explica):
- 'logistica': regresión logística (prediccion/modelos/regresion_logistica.py). La más sencilla y explicable.
- 'xgboost': XGBoost (prediccion/modelos/xgboost_arboles.py). Capta combinaciones; el mejor clásico en TUSZ.
Para añadir un modelo: un fichero nuevo en prediccion/modelos/ y una línea en MODELOS.

Variante "mismo peso por paciente"
==================================
F03.01 (§3.1) vio que en 🔵 train un paciente aporta casi el 10 % de las ventanas. Con la variante
'mismo_peso_paciente' cada ventana pesa 1 / (ventanas de su paciente): todos los pacientes cuentan lo mismo al
aprender, tengan 1 crisis o 30. Es la versión clásica del WeightedRandomSampler que se probará con la red.
"""

import itertools

import numpy as np

from tfm_comun.prediccion.modelos import regresion_logistica, xgboost_arboles

MODELOS = {'logistica': regresion_logistica, 'xgboost': xgboost_arboles}

# Hiperparámetros que se prueban (rejilla pequeña: se elige el mejor con 🟡 dev). Nombre bonito para las tablas.
NOMBRES_MODELO = {nombre: modulo.NOMBRE for nombre, modulo in MODELOS.items()}
REJILLAS = {nombre: modulo.REJILLA for nombre, modulo in MODELOS.items()}
VARIANTES_PESO = {'normal': 'cada ventana pesa lo mismo',
                  'mismo_peso_paciente': 'cada paciente pesa lo mismo (1 / sus ventanas)'}


def combinaciones(rejilla):
    """Todas las combinaciones de una rejilla: {'C': [0.1, 1]} -> [{'C': 0.1}, {'C': 1}]."""
    claves = list(rejilla)
    return [dict(zip(claves, valores)) for valores in itertools.product(*(rejilla[c] for c in claves))]


def crear_modelo(nombre, hiperparametros, semilla=42):
    """Un modelo sin entrenar.

    Parámetros:
        nombre: 'logistica' o 'xgboost' (MODELOS).  hiperparametros: una combinación de REJILLAS[nombre].
        semilla: fija el azar (XGBoost sortea ventanas y características en cada árbol).
    """
    if nombre not in MODELOS:
        raise ValueError(f'Modelo desconocido: {nombre}. Opciones: {list(NOMBRES_MODELO)}')
    return MODELOS[nombre].crear(hiperparametros, semilla)


def pesos_por_paciente(pacientes):
    """Peso de cada ventana para que todos los pacientes sumen lo mismo: 1 / (ventanas de su paciente).

    Se reescalan para que la media sea 1 (así el modelo "ve" el mismo número de ventanas que sin pesos).
    Ejemplo: un paciente con 100 ventanas y otro con 10 -> cada ventana del primero pesa 10 veces menos.
    """
    _, codigo = np.unique(np.asarray(pacientes), return_inverse=True)   # número de paciente de cada ventana
    peso = 1.0 / np.bincount(codigo)[codigo]
    return peso / peso.mean()


def entrenar(nombre, hiperparametros, X, y, pacientes=None, variante='normal', semilla=42):
    """Entrena un modelo y lo devuelve.

    Parámetros:
        X: características (ventanas x 110).  y: 1 = ⏳ preictal, 0 = 💤 interictal.
        pacientes: paciente de cada ventana (hace falta para la variante 'mismo_peso_paciente').
        variante: 'normal' o 'mismo_peso_paciente' (VARIANTES_PESO).
    """
    assert variante in VARIANTES_PESO, f'Variante desconocida: {variante}. Opciones: {list(VARIANTES_PESO)}'
    modelo = crear_modelo(nombre, hiperparametros, semilla)
    peso = pesos_por_paciente(pacientes) if variante == 'mismo_peso_paciente' else None
    modelo.fit(X, y, **({MODELOS[nombre].CLAVE_PESO: peso} if peso is not None else {}))
    return modelo


def puntuar(modelo, X):
    """Puntuación de cada ventana entre 0 y 1 (más alta = más probable ⏳ preictal)."""
    return modelo.predict_proba(X)[:, 1]


def pesos_explicativos(nombre, modelo):
    """Cuánto usa el modelo cada característica, para dibujarlo (mismo orden que las características).

    - Regresión logística: el peso de cada característica, CON signo (ya están en la misma escala).
      Positivo = cuanto más alta, más ⏳ preictal; negativo = más 💤 interictal.
    - XGBoost: la "ganancia" de cada característica (cuánto mejoran los árboles al usarla), sin signo,
      repartida para que sume 1.
    """
    return MODELOS[nombre].pesos_explicativos(modelo)
