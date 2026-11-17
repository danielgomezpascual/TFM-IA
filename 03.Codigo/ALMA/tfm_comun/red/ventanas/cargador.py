"""Cargador (DataLoader): sirve los lotes de una vista, en orden o barajados, de forma repetible.

Motivo para cambiar: que cambie cómo se reparten las ventanas en lotes (tamaño, barajado, procesos en paralelo).

Imagen: fase_03/img/lotes_y_epocas.png (qué es un lote y qué es una época)

Recibe sus dos colaboradores ya hechos (inyección de dependencias): el Dataset (qué ventanas hay) y el preparador
del lote (cómo se convierten en tensores). Así el cargador no sabe nada de normalizaciones ni de aumentos.
"""

import torch
from torch.utils.data import DataLoader

from tfm_comun.red.reproducibilidad import iniciar_trabajador


def crear_cargador(dataset, preparar, tamano_lote=64, barajar=False, semilla=42, n_trabajadores=0, pin_memory=False):
    """El DataLoader que sirve los lotes de una vista.

    Parámetros:
        dataset: un VentanasEEG.  preparar: un PrepararLote (normalización, recorte, aumento).
        tamano_lote: ventanas por lote.
        barajar: True para entrenar (orden distinto en cada época, pero repetible con la semilla); False para
            validar y evaluar (mismo orden que el índice, para poder volver a juntar las predicciones con sus ventanas).
        semilla: fija el barajado y el aumento: con la misma semilla, las mismas épocas.
        n_trabajadores: procesos que preparan lotes en paralelo mientras la red entrena. 0 = en el propio notebook
            (lo más seguro en Windows y en Jupyter); con el paquete en disco y la normalización por lote suele bastar.
        pin_memory: True si se entrena en GPU NVIDIA (copia algo más rápido a la tarjeta).
    """
    generador = torch.Generator().manual_seed(semilla)
    return DataLoader(dataset, batch_size=tamano_lote, shuffle=barajar, collate_fn=preparar, generator=generador,
                      num_workers=n_trabajadores, worker_init_fn=iniciar_trabajador if n_trabajadores else None,
                      persistent_workers=n_trabajadores > 0, pin_memory=pin_memory)
