"""Punto de control: guardar el estado de un entrenamiento de forma segura, para poder parar y seguir.

Motivo para cambiar: que cambie cómo se guarda en disco un punto de control.

Por qué en dos pasos: primero se guarda en un fichero temporal y después se renombra. Renombrar es instantáneo, así
que si el ordenador se corta a mitad de guardar, el punto de control anterior sigue entero (nunca queda uno roto).
Lo usan el entrenamiento de la red (red/entreno/entrenar_red.py) y el preentrenamiento (red/autosupervisado/).
"""

from pathlib import Path

import torch


def guardar_punto_control(ruta, contenido):
    """Guarda un punto de control: primero en un temporal y luego se renombra (un corte a mitad no deja el fichero roto)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.stem + '_tmp.pt')
    torch.save(contenido, tmp)
    tmp.replace(ruta)
