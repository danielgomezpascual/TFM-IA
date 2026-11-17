"""Orden de cada época: qué filas de entrenamiento entran en una época y en qué orden.

Motivo para cambiar: que cambie cómo se sortean las filas de cada época.

Imagen: fase_03/img/secuencias_repetidas.png (por qué se coge una sola fila de cada trozo de 16 s)

Tres piezas que se combinan:
    - orden_de_epoca: baraja (o sortea con pesos, o una fila por grupo) las filas de entrenamiento.
    - peso_por_paciente: el peso que hace que cada paciente cuente lo mismo, tenga 10 filas o 1.000.
    - grupos_espaciados: agrupa las filas casi repetidas (clips que comparten casi todo su minuto).
"""

import numpy as np
import pandas as pd
import torch


def orden_de_epoca(posiciones, generador, pesos=None, grupos=None):
    """En qué orden se ven las filas de entrenamiento en una época.

    Parámetros:
        posiciones: filas de entrenamiento.  generador: torch.Generator (con semilla: épocas repetibles).
        pesos: None = todas una vez, barajadas. Si se da (un peso por fila), se sortean tantas filas como hay, CON
            repetición y con probabilidad proporcional al peso: es lo que hace WeightedRandomSampler de PyTorch.
            Con peso = 1 / (filas de su paciente), cada paciente sale, de media, las mismas veces.
        grupos: None = todas las filas. Si se da (un grupo por fila, p. ej. de grupos_espaciados()), en cada época
            entra UNA fila de cada grupo, sorteada (otra distinta en la época siguiente). Se aplica antes que los pesos.

    Devuelve:
        Array de posiciones, en el orden en que se ven.
    """
    posiciones = np.asarray(posiciones)
    if grupos is not None:
        grupos = np.asarray(grupos)
        azar = torch.rand(len(posiciones), generator=generador).numpy()
        orden = np.lexsort((azar, grupos))                                   # por grupo y, dentro, al azar
        primeras = orden[np.r_[True, grupos[orden][1:] != grupos[orden][:-1]]]
        elegidas = np.sort(primeras)
        posiciones = posiciones[elegidas]
        pesos = None if pesos is None else np.asarray(pesos)[elegidas]
    if pesos is None:
        return posiciones[torch.randperm(len(posiciones), generator=generador).numpy()]
    p = torch.as_tensor(np.asarray(pesos, dtype=np.float64))
    return posiciones[torch.multinomial(p, len(posiciones), replacement=True, generator=generador).numpy()]


def peso_por_paciente(pacientes):
    """1 / (filas de su paciente) para cada fila: el mismo peso total para cada paciente."""
    pacientes = np.asarray(pacientes)
    _, inversa, cuentas = np.unique(pacientes, return_inverse=True, return_counts=True)
    return 1.0 / cuentas[inversa]


def grupos_espaciados(rutas, inicios, separacion_s, fs):
    """Un número de grupo por fila: las filas del mismo tramo cuyas ventanas caen en el mismo trozo de separacion_s.

    Por qué (imagen: fase_03/img/secuencias_repetidas.png): dos ⏳ seguidas empiezan a 2 s una de otra y sus
    secuencias de 60 s comparten 14 de 15 ventanas. Para la red son casi el mismo ejemplo repetido muchas veces, y
    se los aprende de memoria. Con orden_de_epoca(..., grupos=...) entra una sola fila de cada trozo por época
    (sorteada), así que en una época no hay dos secuencias casi iguales; en la siguiente entra otra del trozo.

    Ejemplo: separacion_s = 16 -> de cada 16 s de un tramo entra una fila por época (de las 8 ⏳ o 4 💤 que hay).
    """
    trozo = (np.asarray(inicios) // int(round(separacion_s * fs))).astype(np.int64)
    claves = pd.Series(np.asarray(rutas)).astype(str) + '|' + pd.Series(trozo).astype(str)
    return pd.factorize(claves)[0]
