"""Transferencia de pesos (F03.07): pasar lo aprendido sin etiquetas a la red de predicción.

Motivo para cambiar: que cambie qué piezas se copian o cuáles se dejan fijas.

    pasar_pesos          -> copia el CODIFICADOR preentrenado (proyección + GRU) a una RedGRUSegundos nueva; la capa
                            final (la que da el logit de ⏳) queda sin entrenar, porque la tarea es otra.
    congelar_codificador -> deja fijos esos pesos: solo aprende la capa final. Responde a "¿lo aprendido sirve tal cual?".
"""

import torch


def pasar_pesos(ruta_mejor, red, dispositivo):
    """Copia el codificador preentrenado (proyección + GRU) a una RedGRUSegundos nueva (red/redes/gru_segundos.py). La capa final queda sin entrenar.

    Devuelve:
        Lista de las piezas copiadas (para comprobarlo).
    """
    punto = torch.load(ruta_mejor, map_location=dispositivo, weights_only=False)
    codificador = {k[len('red.'):]: v for k, v in punto['modelo'].items()
                   if k.startswith('red.') and not k.startswith('red.final.')}
    faltan, sobran = red.load_state_dict(codificador, strict=False)
    assert not sobran, f'Piezas que no encajan: {sobran}'
    assert all(k.startswith('final.') for k in faltan), f'Falta copiar: {faltan}'
    return sorted({k.split('.')[0] for k in codificador})


def congelar_codificador(red):
    """Deja fijos la proyección y la GRU: solo aprende la capa final (¿lo aprendido ya sirve tal cual?)."""
    for nombre, parametro in red.named_parameters():
        parametro.requires_grad = nombre.startswith('final.')
    return red
