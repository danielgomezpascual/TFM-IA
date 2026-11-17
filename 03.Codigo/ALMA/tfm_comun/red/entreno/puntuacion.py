"""Puntuación: lo que dice una red ya entrenada sobre unas filas (su probabilidad de ⏳ y su pérdida media).

Motivo para cambiar: que cambie cómo se pasa la red por unas filas sin entrenarla.

Modo evaluación (modelo.eval()): el dropout deja de apagar conexiones al azar y BatchNorm usa sus estadísticas
fijas. Con torch.no_grad() PyTorch no guarda lo necesario para los gradientes: va más rápido y gasta menos memoria.
"""

import numpy as np
import torch


@torch.no_grad()
def puntuar_red(modelo, datos, posiciones, k=None, tamano_lote=1024):
    """La probabilidad de ⏳ que da la red a cada fila (numpy, en el orden de `posiciones`).

    Pone la red en modo evaluación (sin dropout y con las estadísticas fijas de BatchNorm) y no guarda gradientes.
    """
    modelo.eval()
    posiciones = np.asarray(posiciones)
    salida = []
    for a in range(0, len(posiciones), tamano_lote):
        x, _ = datos.lote(posiciones[a:a + tamano_lote], k=k)
        salida.append(torch.sigmoid(modelo(x)).float().cpu().numpy())
    return np.concatenate(salida) if salida else np.empty(0, dtype=np.float32)


def perdida_media(modelo, datos, posiciones, k, tamano_lote, funcion):
    """La pérdida media de la red en unas filas, sin entrenar (modo evaluación y sin gradientes).

    Parámetros:
        modelo: la red.  datos: SecuenciasEspectro (red/lotes/secuencias_espectro.py).  posiciones: filas.
        k: ventanas de la secuencia que se usan.  tamano_lote: filas por lote.
        funcion: la pérdida (p. ej. nn.BCEWithLogitsLoss()).
    """
    modelo.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for a in range(0, len(posiciones), tamano_lote):
            x, y = datos.lote(posiciones[a:a + tamano_lote], k=k)
            total += float(funcion(modelo(x), y)) * len(y)
            n += len(y)
    return total / max(n, 1)
