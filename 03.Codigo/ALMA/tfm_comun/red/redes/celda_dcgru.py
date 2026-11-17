"""Celda DCGRU (CeldaDCGRU, F03.08): una GRU en la que cada multiplicación es una convolución de difusión (Li et al., 2018).

Motivo para cambiar: que cambie cómo se actualiza la memoria de cada canal en cada segundo.

Imagen: fase_03/img/gru_por_dentro.png (las dos puertas de una GRU; aquí, con vecinos)

Igual que una GRU normal (F03.04 §6), con dos puertas:
    r (reinicio):      cuánto de la memoria anterior se usa para proponer la nueva;
    u (actualización): cuánto se conserva de la memoria anterior frente a la propuesta;
pero cada canal lleva SU memoria (lote, nodos, memoria) y, al calcular las puertas y la propuesta, mezcla la suya
con la de sus vecinos del grafo (ConvolucionDifusion, red/redes/convolucion_difusion.py).
"""

import torch
from torch import nn

from tfm_comun.red.redes.convolucion_difusion import ConvolucionDifusion


class CeldaDCGRU(nn.Module):
    """Una GRU con convoluciones de difusión (Li et al., 2018): memoria (lote, nodos, memoria)."""

    def __init__(self, entradas, memoria, k_difusion=2):
        super().__init__()
        self.memoria = memoria
        self.puertas = ConvolucionDifusion(entradas + memoria, 2 * memoria, k_difusion)
        self.candidata = ConvolucionDifusion(entradas + memoria, memoria, k_difusion)

    def forward(self, x, h, matrices):
        r, u = torch.sigmoid(self.puertas(torch.cat([x, h], dim=-1), matrices)).chunk(2, dim=-1)
        c = torch.tanh(self.candidata(torch.cat([x, r * h], dim=-1), matrices))
        return u * h + (1 - u) * c
