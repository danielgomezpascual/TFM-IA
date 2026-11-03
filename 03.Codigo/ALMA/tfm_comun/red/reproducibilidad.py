"""
Reproducibilidad: que entrenar dos veces con la misma configuración dé (casi) lo mismo.

Motivo para cambiar: que cambie cómo se fijan las semillas del azar.

En una red hay azar en muchos sitios: los pesos iniciales, el orden de las ventanas en cada época, el aumento
de datos, el dropout... Fijando la semilla de cada generador (Python, numpy y PyTorch) el resultado se repite.
En GPU puede variar en los últimos decimales (algunas operaciones de CUDA no son deterministas), lo normal.
"""

import random

import numpy as np
import torch


def fijar_semillas(semilla=42):
    """Fija la semilla de Python, numpy y PyTorch (también en todas las GPU)."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def iniciar_trabajador(id_trabajador):
    """Semilla de cada proceso que ayuda al DataLoader (worker_init_fn).

    PyTorch ya da a cada trabajador una semilla distinta para torch; esta función usa esa misma para numpy
    y random, que si no saldrían iguales en todos los trabajadores (y repetirían el mismo "azar").
    """
    semilla = torch.initial_seed() % 2**32
    np.random.seed(semilla)
    random.seed(semilla)
