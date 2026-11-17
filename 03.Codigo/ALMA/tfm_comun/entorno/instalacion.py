"""Instalación: instalar con pip, en el MISMO Python del notebook, las librerías que falten (y PyTorch con GPU si se puede).

Motivo para cambiar: que cambie cómo se instala una librería o de dónde se baja PyTorch.

Por qué sys.executable: es la ruta exacta del Python que ejecuta el notebook (el del .venv). Llamar a "pip" a secas
podría instalar en otro Python del ordenador y el notebook seguiría sin verla.
"""

import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys

from tfm_comun.entorno.gpu_nvidia import cuda_del_driver, hay_gpu_nvidia, torch_usa_gpu


# PyTorch para tarjetas NVIDIA no está en PyPI (allí solo está la versión para CPU en Windows): está en el
# repositorio de PyTorch, una carpeta por versión de CUDA. {versión de CUDA: repositorio}, de la más nueva a la más antigua.
INDICES_PYTORCH_CUDA = {(13, 2): 'https://download.pytorch.org/whl/cu132',
                        (13, 0): 'https://download.pytorch.org/whl/cu130',
                        (12, 8): 'https://download.pytorch.org/whl/cu128',
                        (12, 6): 'https://download.pytorch.org/whl/cu126'}


def version_instalada(nombre_pip):
    """Versión instalada de una librería, o None si no está. Ignora los extras: 'paquete[jupyter]' -> 'paquete'."""
    try:
        return importlib.metadata.version(nombre_pip.split('[')[0])
    except importlib.metadata.PackageNotFoundError:
        return None


def instalada(modulo):
    """True si la librería se puede importar."""
    return importlib.util.find_spec(modulo) is not None


def instalar(nombres_pip):
    """Instala librerías con pip en el MISMO Python que ejecuta el notebook (sys.executable = el del .venv).

    Después vacía la caché de importación para que Python vea lo nuevo sin reiniciar el kernel.
    Si pip falla, enseña el final de su mensaje de error.
    """
    if 'torch' in nombres_pip:                          # PyTorch va aparte: puede necesitar el repositorio de CUDA
        instalar_torch()
        nombres_pip = [n for n in nombres_pip if n != 'torch']
        if not nombres_pip:
            return
    print(f'Instalando: {" ".join(nombres_pip)} (puede tardar un par de minutos)...')
    resultado = subprocess.run([sys.executable, '-m', 'pip', 'install', *nombres_pip], capture_output=True, text=True)
    importlib.invalidate_caches()
    if resultado.returncode != 0:
        print(resultado.stderr[-2000:])


def instalar_torch():
    """Instala PyTorch: con soporte de GPU si hay una tarjeta NVIDIA, si no la versión para CPU.

    Con NVIDIA:
        1. Lee qué CUDA admite el driver (cuda_del_driver) y solo prueba repositorios de esa CUDA o anteriores.
        2. Instala desde el primero y comprueba que PyTorch VE la GPU (torch_usa_gpu). Si no la ve, lo quita y
           prueba el siguiente.
        3. Si ninguno vale, instala la de CPU desde PyPI (funciona igual, pero entrenar es mucho más lento).
    La descarga con CUDA ocupa ~2,5 GB.
    """
    if hay_gpu_nvidia():
        driver = cuda_del_driver()
        candidatos = [(v, i) for v, i in INDICES_PYTORCH_CUDA.items() if driver is None or v <= driver]
        print(f'Tarjeta NVIDIA detectada (el driver admite CUDA {".".join(map(str, driver)) if driver else "?"}).')
        for version, indice in candidatos:
            print(f'Instalando PyTorch con CUDA {version[0]}.{version[1]} desde {indice} (~2,5 GB, puede tardar)...')
            resultado = subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', '--index-url', indice],
                                       capture_output=True, text=True)
            importlib.invalidate_caches()
            if resultado.returncode == 0:
                ok, texto = torch_usa_gpu()
                if ok:
                    print(f'✅ PyTorch ve la GPU: {texto}')
                    return
                print(f'   Instalado, pero no ve la GPU ({texto}); se prueba el siguiente.')
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch'], capture_output=True, text=True)
            else:
                print(f'   No hay PyTorch para este Python en {indice}; se prueba el siguiente.')
        print('🔴 No se ha podido instalar PyTorch con CUDA: se instala la versión para CPU.')
    else:
        print('No hay tarjeta NVIDIA: se instala PyTorch para CPU.')
    resultado = subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch'], capture_output=True, text=True)
    importlib.invalidate_caches()
    if resultado.returncode != 0:
        print(resultado.stderr[-2000:])


def reinstalar_torch_gpu():
    """Si hay NVIDIA y el PyTorch instalado no ve la GPU (p. ej. es la versión +cpu), lo cambia por uno con CUDA.

    Devuelve True si ha reinstalado (entonces hay que reiniciar el kernel si el notebook ya había importado torch).
    """
    if not (hay_gpu_nvidia() and instalada('torch')):
        return False
    ok, texto = torch_usa_gpu()
    if ok:
        return False
    print(f'🔴 Hay una tarjeta NVIDIA, pero este PyTorch no la usa ({texto}). Se cambia por la versión con CUDA.')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch'], capture_output=True, text=True)
    instalar_torch()
    return True
