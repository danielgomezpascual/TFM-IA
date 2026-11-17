"""
Red de preentrenamiento (F03.07): aprende "cómo es un EEG" prediciendo los segundos siguientes, sin etiquetas.

Motivo para cambiar: que cambie la arquitectura de la red de preentrenamiento (la tarea y el entrenamiento están en
red/autosupervisado/).

La red de preentrenamiento (RedPreentreno)
==========================================
    codificador = la proyección + la GRU de RedGRUSegundos (lo que luego se reutiliza)
    decodificador = otra GRU que arranca con la memoria del codificador y da 12 pasos (uno por segundo futuro)
    salida = capa lineal: cada paso -> los 22 canales x 40 Hz de ese segundo
"""

import torch
from torch import nn

from tfm_comun.red.redes.gru_segundos import RedGRUSegundos


class RedPreentreno(nn.Module):
    """Codificador de RedGRUSegundos + decodificador que predice los segundos siguientes."""

    def __init__(self, n_canales, n_frecuencias, arquitectura, segundos_futuro=12):
        super().__init__()
        arquitectura = {k: v for k, v in arquitectura.items() if k != 'tipo'}
        self.red = RedGRUSegundos(n_canales, n_frecuencias, **arquitectura)
        memoria = self.red.gru.hidden_size
        self.segundos_futuro = segundos_futuro
        self.decodificador = nn.GRU(memoria, memoria, num_layers=self.red.gru.num_layers, batch_first=True)
        self.salida = nn.Linear(memoria, n_canales * self.red.frecuencias)

    def forward(self, x):
        """x: (lote, k, canales, segundos x frecuencias) -> predicción (lote, segundos_futuro, canales x frecuencias)."""
        _, memoria = self.red.gru(self.red.proyeccion(self.red.secuencia_de_segundos(x)))
        entrada = torch.zeros(x.shape[0], self.segundos_futuro, memoria.shape[-1], device=x.device)
        salida, _ = self.decodificador(entrada, memoria)
        return self.salida(salida)
