"""Preparar el lote (PrepararLote): de una lista de ventanas sueltas al tensor que entra en la red.

Motivo para cambiar: que cambie cómo se normaliza, se recorta o se aumenta un lote.

Es la collate_fn del DataLoader: PyTorch le pasa las (ventana, etiqueta) de un lote y ella devuelve (x, y).
Se hace por lote y no por ventana porque numpy normaliza 64 ventanas a la vez casi tan rápido como una.

Piezas que usa (cada una en su sitio):
    - la normalización por ventana y canal: senal/normalizacion.py (NORMALIZACIONES);
    - el reflejo izquierda <-> derecha: red/aumentos/reflejo_hemisferios.py.
"""

import numpy as np
import torch

from tfm_comun.red.aumentos.reflejo_hemisferios import reflejar_hemisferios
from tfm_comun.senal.normalizacion import NORMALIZACIONES


class PrepararLote:
    """Convierte una lista de (ventana, etiqueta) en el lote que entra en la red (collate_fn del DataLoader).

    Qué hace, en orden:
        1. Junta las ventanas: (lote, 22, 1.024) en float16.
        2. Normaliza cada ventana y canal con `normalizacion` ('robusta' o 'zscore', senal/normalizacion.py).
        3. Recorta a ±recorte (si se da): un artefacto enorme no se queda en cientos de "desviaciones" y no
           desestabiliza el entrenamiento.
        4. Si `reflejo` > 0, refleja izquierda <-> derecha esa fracción de ventanas, al azar (solo al entrenar).

    Devuelve:
        (x, y): x tensor float32 (lote, 22, 1.024); y tensor float32 (lote,) con 1 = ⏳ preictal, 0 = 💤 interictal.
    """

    def __init__(self, normalizacion='robusta', recorte=None, reflejo=0.0):
        assert normalizacion in NORMALIZACIONES, f'Normalización desconocida: {normalizacion}. Opciones: {list(NORMALIZACIONES)}'
        self.normalizacion, self.recorte, self.reflejo = normalizacion, recorte, reflejo

    def __call__(self, muestras):
        ventanas = np.stack([v for v, _ in muestras])
        ventanas = NORMALIZACIONES[self.normalizacion](ventanas).astype(np.float32, copy=False)   # la red trabaja en float32
        if self.recorte:
            np.clip(ventanas, -self.recorte, self.recorte, out=ventanas)
        x = torch.from_numpy(ventanas)
        if self.reflejo > 0:
            x = reflejar_hemisferios(x, self.reflejo)
        y = torch.as_tensor(np.array([e for _, e in muestras], dtype=np.float32))
        return x, y
