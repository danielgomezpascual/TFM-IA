"""Convolución de difusión (ConvolucionDifusion, F03.08): la operación básica de la red de grafos de Tang (DCRNN).

Motivo para cambiar: que cambie cómo un nodo mezcla su información con la de sus vecinos.

Imagen: fase_03/img/grafo_dcrnn.png (el grafo de un clip y la difusión a 1 y 2 pasos)

En sencillo: "lo mío + lo de mis vecinos + lo de los vecinos de mis vecinos", con pesos que se aprenden.
    1. Se parte de X (lote, nodos, números): lo que tiene cada canal.
    2. P·X es "lo que llega de los vecinos a 1 paso" (P = matriz de paso, red/grafo.py: cada fila suma 1, es una
       media ponderada de los vecinos). P·P·X, a 2 pasos. Se hace con las dos direcciones del grafo (P₁ y P₂).
    3. Se juntan [X, P₁X, P₁²X, P₂X, P₂²X] y una capa lineal, LA MISMA para todos los nodos, da la salida.
"""

import torch
from torch import nn


class ConvolucionDifusion(nn.Module):
    """Convolución de difusión: cada nodo combina lo suyo con lo de sus vecinos a 1..K pasos, en las dos direcciones."""

    def __init__(self, entradas, salidas, k_difusion=2, n_matrices=2):
        super().__init__()
        self.k_difusion = k_difusion
        self.lineal = nn.Linear(entradas * (1 + n_matrices * k_difusion), salidas)

    def forward(self, x, matrices):                      # x: (lote, nodos, entradas)
        partes = [x]
        for paso in matrices:
            actual = x
            for _ in range(self.k_difusion):
                actual = torch.bmm(paso, actual)
                partes.append(actual)
        return self.lineal(torch.cat(partes, dim=-1))
