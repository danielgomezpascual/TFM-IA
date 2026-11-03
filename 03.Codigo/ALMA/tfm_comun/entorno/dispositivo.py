"""
Dispositivo: dónde se hacen las cuentas de la red (tarjeta gráfica o procesador).

Motivo para cambiar: que cambie cómo se elige el dispositivo o qué se cuenta de él.

Una red se entrena multiplicando matrices enormes. Una tarjeta gráfica (GPU) hace miles de esas
operaciones a la vez y entrena del orden de 10-50 veces más rápido que el procesador (CPU).
PyTorch usa la GPU si:
    - es NVIDIA (CUDA) y PyTorch se instaló con CUDA (requisitos.py lo hace solo), o
    - es un Mac con chip Apple (MPS).
Si no, usa la CPU: funciona igual, pero más despacio.
"""

import platform

import torch

from tfm_comun.entorno.requisitos import hay_gpu_nvidia


def elegir_dispositivo():
    """El mejor dispositivo disponible y una ficha con sus datos.

    Orden: GPU NVIDIA (cuda) -> GPU de Apple (mps) -> procesador (cpu).

    Devuelve:
        (dispositivo, ficha):
            dispositivo: torch.device, para hacer modelo.to(dispositivo) y lote.to(dispositivo).
            ficha: diccionario {'dispositivo', 'nombre', 'memoria (GB)', 'aviso'} para enseñarlo en una tabla.
                'aviso' explica si hay una GPU NVIDIA que PyTorch no está usando (instalado sin CUDA).
    """
    if torch.cuda.is_available():
        propiedades = torch.cuda.get_device_properties(0)
        return torch.device('cuda'), {'dispositivo': 'cuda (GPU NVIDIA)', 'nombre': propiedades.name,
                                      'memoria (GB)': round(propiedades.total_memory / 1e9, 1), 'aviso': ''}
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps'), {'dispositivo': 'mps (GPU de Apple)', 'nombre': platform.processor() or 'Apple',
                                     'memoria (GB)': None, 'aviso': ''}
    aviso = ('🔴 Hay una tarjeta NVIDIA, pero este PyTorch no tiene CUDA (es la versión para CPU). Para usar la GPU: '
             'pip uninstall torch y volver a ejecutar la celda de requisitos, que instala la versión con CUDA.'
             if hay_gpu_nvidia() else 'Sin GPU: se entrenará en el procesador (más lento).')
    return torch.device('cpu'), {'dispositivo': 'cpu (procesador)', 'nombre': platform.processor() or platform.machine(),
                                 'memoria (GB)': None, 'aviso': aviso}
