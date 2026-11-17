"""Codificador de ventana (CodificadorVentana, F03.04): la CNN que resume el espectro de UNA ventana de 4 s.

Motivo para cambiar: que cambie cómo se resume una ventana (capas, filtros, tamaño del resumen).

Imagen: fase_03/img/arquitectura_cnn_gru.png (la pieza "CNN" de la red)

Qué hace una convolución 1D aquí: un filtro de 3 Hz recorre el eje de la frecuencia y, en cada posición, combina
los 22 canales en esas 3 frecuencias. Aprende formas como "pico en 8-12 Hz" o "mucha energía lenta". Con varios
filtros (32) aprende varias formas a la vez. Es una pieza independiente: RedCNNGRU la usa para cada una de las 15
ventanas, con los MISMOS pesos (una ventana se resume igual esté donde esté).
"""

import torch
from torch import nn


class CodificadorVentana(nn.Module):
    """CNN que resume el espectro de una ventana (canales x frecuencias) en `resumen` números.

    Capas (con 22 canales y 40 frecuencias):
        Conv1d 22 -> filtros, filtro de 3 Hz  + BatchNorm + GELU + MaxPool /2   -> (filtros, 20)
        Conv1d filtros -> filtros, 3 Hz        + BatchNorm + GELU + MaxPool /2   -> (filtros, 10)
        aplanar (filtros x 10) -> Dropout -> Linear -> resumen  + GELU
    Se aplana en vez de promediar las frecuencias para no perder DÓNDE está la energía (no es lo mismo 3 Hz que 30 Hz).
    BatchNorm: reajusta la escala de cada filtro en cada lote (entrena más estable). GELU: la "no linealidad"
    (sin ella, varias capas equivaldrían a una sola).
    """

    def __init__(self, n_canales=22, n_frecuencias=40, filtros=32, resumen=64, dropout=0.3):
        super().__init__()
        self.convoluciones = nn.Sequential(
            nn.Conv1d(n_canales, filtros, kernel_size=3, padding=1), nn.BatchNorm1d(filtros), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(filtros, filtros, kernel_size=3, padding=1), nn.BatchNorm1d(filtros), nn.GELU(), nn.MaxPool1d(2),
        )
        with torch.no_grad():
            n_salida = self.convoluciones(torch.zeros(2, n_canales, n_frecuencias)).numel() // 2
        self.salida = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n_salida, resumen), nn.GELU())

    def forward(self, x):                      # x: (ventanas, canales, frecuencias)
        return self.salida(self.convoluciones(x))
