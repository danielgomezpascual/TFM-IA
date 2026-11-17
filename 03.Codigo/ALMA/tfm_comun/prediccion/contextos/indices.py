"""Índices del contexto: dónde está, en el almacén, cada una de las k ventanas del contexto de cada fila.

Motivo para cambiar: que cambie cómo se localiza el contexto de una fila en el almacén.

Sirve para no copiar datos: cada ventana calculada se guarda UNA vez y la secuencia de cada fila se monta con estos
índices (dos ⏳ seguidas comparten 14 de sus 15 ventanas).
"""

import numpy as np
import pandas as pd

from tfm_comun.prediccion.contextos.disponibilidad import clave


def indices_contexto(filas, k, tamano, claves_calculadas):
    """Dónde está, en el almacén, cada una de las k ventanas del contexto de cada fila: array int64 (filas, k).

    La columna 0 es la ventana más antigua y la k - 1, la actual. Sirve para no copiar datos: la red de F03.04
    guarda cada ventana calculada UNA vez y monta la secuencia de cada fila con estos índices al hacer el lote.
    claves_calculadas: clave() de cada ventana del almacén, en su orden.
    """
    indice = pd.Index(claves_calculadas)
    columnas = []
    for j in range(k - 1, -1, -1):                       # de la más antigua a la actual
        posiciones = indice.get_indexer(clave(filas.ruta.to_numpy(), filas.muestra_inicio.to_numpy() - j * tamano))
        assert (posiciones >= 0).all(), 'Faltan características de alguna ventana del contexto.'
        columnas.append(posiciones)
    return np.stack(columnas, axis=1).astype(np.int64)


def cubo_contexto(filas, k, tamano, claves_calculadas, caracteristicas):
    """Las características de las k ventanas del contexto de cada fila: array (filas, k, 110).

    La posición 0 es la ventana más antigua y la k - 1, la actual.
    claves_calculadas / caracteristicas: lo que devuelve el almacén (clave() de cada ventana calculada y sus 110 números).
    """
    return caracteristicas[indices_contexto(filas, k, tamano, claves_calculadas)]
