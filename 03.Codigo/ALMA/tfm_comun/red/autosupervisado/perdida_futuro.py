"""Pérdida del preentrenamiento (F03.07): cuánto se equivoca la red al predecir los segundos siguientes.

Motivo para cambiar: que cambie cómo se mide el error de la predicción del futuro.

    mae_con_mascara      -> el error absoluto medio en los números que existen (Tang usa el mismo).
    perdida_media_futuro -> esa pérdida en unas secuencias, sin entrenar (para validar).
    perdida_sin_aprender -> el SUELO: el error de decir siempre "el futuro será lo normal del paciente" (todo 0 tras
                            normalizar). Si la red no baja de aquí, no ha aprendido nada.
"""

import torch


def mae_con_mascara(prediccion, objetivo, mascara):
    """Error absoluto medio (MAE) solo en los números que existen (mascara = True). Es la pérdida de Tang."""
    mascara = mascara.float()
    return (torch.abs(prediccion - objetivo) * mascara).sum() / mascara.sum().clamp(min=1)


@torch.no_grad()
def perdida_media_futuro(modelo, lotes, inicios, tamano_lote):
    """La pérdida media (MAE) de la red al predecir el futuro de unas secuencias, sin entrenar."""
    modelo.eval()
    total, n = 0.0, 0
    for a in range(0, len(inicios), tamano_lote):
        x, y, m = lotes.lote(inicios[a:a + tamano_lote])
        total += float(mae_con_mascara(modelo(x), y, m)) * len(x)
        n += len(x)
    return total / max(n, 1)


def perdida_sin_aprender(lotes, inicios, tamano_lote=1024):
    """La pérdida de "predecir que el futuro es lo normal del paciente" (todo 0 tras normalizar): el suelo a batir."""
    total, n = 0.0, 0
    for a in range(0, len(inicios), tamano_lote):
        _, y, m = lotes.lote(inicios[a:a + tamano_lote])
        total += float(mae_con_mascara(torch.zeros_like(y), y, m)) * len(y)
        n += len(y)
    return total / max(n, 1)
