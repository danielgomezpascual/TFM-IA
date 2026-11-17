"""GPU NVIDIA: ¿hay tarjeta NVIDIA, qué CUDA admite su driver y si el PyTorch instalado la ve?

Motivo para cambiar: que cambie cómo se detecta la tarjeta o su versión de CUDA.

CUDA es el "idioma" con el que PyTorch habla con la tarjeta NVIDIA. Cada driver admite CUDA hasta una versión: un
PyTorch hecho para una CUDA más nueva se instala, pero no ve la GPU. Por eso, antes de instalar PyTorch, se mira qué
CUDA admite el driver (nvidia-smi lo dice en su cabecera: "CUDA Version: 13.2").
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path


# Dónde suele estar nvidia-smi en Windows si no está en el PATH
RUTAS_NVIDIA_SMI = [r'C:\Windows\System32\nvidia-smi.exe', r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe']


def _nvidia_smi():
    """Ruta del programa nvidia-smi (viene con el driver de NVIDIA), o None si no está."""
    encontrado = shutil.which('nvidia-smi')
    if encontrado:
        return encontrado
    return next((r for r in RUTAS_NVIDIA_SMI if Path(r).exists()), None)


def hay_gpu_nvidia():
    """True si el ordenador tiene una tarjeta NVIDIA con su driver (existe el programa nvidia-smi)."""
    return _nvidia_smi() is not None


def cuda_del_driver():
    """Versión máxima de CUDA que admite el driver de NVIDIA, como (13, 2), o None si no se puede saber.

    Sale de la cabecera de nvidia-smi ("CUDA Version: 13.2"). Un PyTorch hecho para una CUDA más nueva que esta
    se instala, pero no ve la GPU: por eso se elige el repositorio según este número.
    """
    ruta = _nvidia_smi()
    if not ruta:
        return None
    salida = subprocess.run([ruta], capture_output=True, text=True).stdout
    encontrado = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', salida)
    return (int(encontrado.group(1)), int(encontrado.group(2))) if encontrado else None


def torch_usa_gpu():
    """¿El PyTorch instalado ve la GPU? Se pregunta en un Python aparte, para que valga aunque el notebook
    ya tenga cargado un PyTorch antiguo. Devuelve (True/False, texto con versión y GPU)."""
    codigo = ('import torch; ok = torch.cuda.is_available(); '
              'print(ok, torch.__version__, torch.cuda.get_device_name(0) if ok else "-", sep="|")')
    resultado = subprocess.run([sys.executable, '-c', codigo], capture_output=True, text=True)
    if resultado.returncode != 0:
        return False, 'PyTorch no se puede importar'
    ok, version, gpu = resultado.stdout.strip().splitlines()[-1].split('|')
    return ok == 'True', f'torch {version} · GPU: {gpu}'
