"""
Requisitos de cada notebook: qué librerías necesita, si están instaladas y, si no, instalarlas.

Motivo para cambiar: que un notebook necesite librerías nuevas o cambie cómo se comprueban o instalan.
Se llama al principio del notebook, ANTES de importar nada más. Si falta alguna librería, la instala
sola con pip en el mismo Python del notebook (el del .venv) y sigue, normalmente sin reiniciar el kernel.

Dos tipos de librería:
    - obligatorias: sin ellas el notebook no funciona -> si no se pueden instalar, se para con un error.
    - opcionales: solo hacen falta para una parte (p. ej. el 3D) -> si no se pueden instalar, aviso en 🔴
      y el notebook sigue (esa parte se salta).

Uso:
    disponibles = comprobar_requisitos(REQUISITOS_03V)
    if disponibles['pyvista']: ...
"""

import importlib
import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

# Cada librería: nombre para importar -> (nombre para pip, para qué se usa, obligatoria)
REQUISITOS_03V = {
    'numpy': ('numpy', 'cálculo con matrices', True),
    'pandas': ('pandas', 'tablas', True),
    'matplotlib': ('matplotlib', 'figuras', True),
    'scipy': ('scipy', 'espectros (Welch) y espectrogramas', True),
    'mne': ('mne', 'leer el EDF, montaje, filtros y localización de fuentes', True),
    'nibabel': ('nibabel', 'leer el cerebro plantilla y su atlas (sección 14)', True),
    'pyvista': ('pyvista', 'dibujar el cerebro en 3D (sección 14)', False),
    'pyvistaqt': ('pyvistaqt', 'motor 3D de MNE (sección 14)', False),
    'PyQt6': ('pyqt6', 'ventanas del 3D (sección 14)', False),
    'trame_pyvista': ('trame-pyvista[jupyter]', 'girar el cerebro 3D dentro del notebook (sección 14)', False),
    'ipywidgets': ('ipywidgets', 'controles de Jupyter (3D interactivo y barras de progreso)', False),
}

REQUISITOS_05_ATAJOS = {
    'numpy': ('numpy', 'cálculo con matrices', True),
    'pandas': ('pandas', 'tablas', True),
    'matplotlib': ('matplotlib', 'figuras', True),
    'sklearn': ('scikit-learn', 'AUROC y modelos sencillos (regresión logística y árboles) con solo metadatos', True),
}

REQUISITOS_FASE_03_DATOS = {
    'numpy': ('numpy', 'cálculo con matrices', True),
    'pandas': ('pandas', 'tablas', True),
    'matplotlib': ('matplotlib', 'figuras', True),
    'tqdm': ('tqdm', 'barra de progreso al crear los paquetes', True),
    'torch': ('torch', 'red neuronal: Dataset, DataLoader y tensores (PyTorch)', True),
}

# PyTorch para tarjetas NVIDIA no está en PyPI (allí solo está la versión para CPU en Windows): está en el
# repositorio de PyTorch, una carpeta por versión de CUDA. {versión de CUDA: repositorio}, de la más nueva a la más antigua.
INDICES_PYTORCH_CUDA = {(13, 2): 'https://download.pytorch.org/whl/cu132',
                        (13, 0): 'https://download.pytorch.org/whl/cu130',
                        (12, 8): 'https://download.pytorch.org/whl/cu128',
                        (12, 6): 'https://download.pytorch.org/whl/cu126'}

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


def _version(nombre_pip):
    """Versión instalada de una librería, o None si no está. Ignora los extras: 'paquete[jupyter]' -> 'paquete'."""
    try:
        return importlib.metadata.version(nombre_pip.split('[')[0])
    except importlib.metadata.PackageNotFoundError:
        return None


def _instalada(modulo):
    """True si la librería se puede importar."""
    return importlib.util.find_spec(modulo) is not None


def _instalar(nombres_pip):
    """Instala librerías con pip en el MISMO Python que ejecuta el notebook (sys.executable = el del .venv).

    Después vacía la caché de importación para que Python vea lo nuevo sin reiniciar el kernel.
    Si pip falla, enseña el final de su mensaje de error.
    """
    if 'torch' in nombres_pip:                          # PyTorch va aparte: puede necesitar el repositorio de CUDA
        _instalar_torch()
        nombres_pip = [n for n in nombres_pip if n != 'torch']
        if not nombres_pip:
            return
    print(f'Instalando: {" ".join(nombres_pip)} (puede tardar un par de minutos)...')
    resultado = subprocess.run([sys.executable, '-m', 'pip', 'install', *nombres_pip], capture_output=True, text=True)
    importlib.invalidate_caches()
    if resultado.returncode != 0:
        print(resultado.stderr[-2000:])


def _instalar_torch():
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
    if not (hay_gpu_nvidia() and _instalada('torch')):
        return False
    ok, texto = torch_usa_gpu()
    if ok:
        return False
    print(f'🔴 Hay una tarjeta NVIDIA, pero este PyTorch no la usa ({texto}). Se cambia por la versión con CUDA.')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch'], capture_output=True, text=True)
    _instalar_torch()
    return True


def comprobar_requisitos(requisitos, instalar=True):
    """Comprueba que están instaladas las librerías del notebook, instala las que falten y lo enseña en una tabla.

    Parámetros:
        requisitos: diccionario como REQUISITOS_03V.
        instalar: si es True (por defecto), instala con pip lo que falte.
    Qué hace:
        1. Mira si cada librería se puede importar.
        2. Si falta alguna y instalar=True, la instala con pip (en el Python del notebook) y vuelve a mirar.
        3. Enseña una tabla: ✅ ya estaba (con versión), 🆕 recién instalada, ❌ falta (obligatoria),
           🔴 falta (opcional).
        4. Si sigue faltando alguna obligatoria, para con un error. Si solo faltan opcionales, avisa y sigue.
    Devuelve: {nombre para importar: True/False}, para saber luego qué partes se pueden ejecutar.
    """
    antes = {modulo: _instalada(modulo) for modulo in requisitos}
    if instalar and not all(antes.values()):
        _instalar([requisitos[m][0] for m, ok in antes.items() if not ok])
    reinstalado = instalar and 'torch' in requisitos and antes.get('torch') and reinstalar_torch_gpu()

    filas, disponibles = [], {}
    for modulo, (nombre_pip, para_que, obligatoria) in requisitos.items():
        disponibles[modulo] = _instalada(modulo)
        if disponibles[modulo]:
            estado = '✅' if antes[modulo] else '🆕 recién instalada'
        else:
            estado = '❌ falta' if obligatoria else '🔴 falta (opcional)'
        filas.append({'librería': nombre_pip, 'estado': estado, 'versión': _version(nombre_pip) or '-',
                      'para qué': para_que})
    display(pd.DataFrame(filas).set_index('librería').style.set_properties(**{'text-align': 'left'}))

    faltan = {m: requisitos[m] for m, ok in disponibles.items() if not ok}
    obligatorias = [pip for pip, _, obligatoria in faltan.values() if obligatoria]
    opcionales = [pip for pip, _, obligatoria in faltan.values() if not obligatoria]
    if opcionales:
        display(Markdown(f'🔴 **No se han podido instalar** {", ".join(opcionales)}: esas partes se saltarán. '
                         f'Revisa el mensaje de pip de arriba.'))
    if obligatorias:
        raise ModuleNotFoundError(f'No se han podido instalar: {", ".join(obligatorias)}. Revisa el mensaje de pip '
                                  f'de arriba o prueba en una celda: %pip install {" ".join(obligatorias)}')
    if reinstalado:
        display(Markdown('🔁 **PyTorch se ha cambiado por la versión con GPU.** Reinicia el kernel y vuelve a ejecutar el '
                         'notebook desde el principio (el PyTorch antiguo sigue cargado en este kernel).'))
    if not faltan:
        print('Todo lo necesario está instalado.' + ('' if all(antes.values()) else
              ' Si una recién instalada diera error al importarla, reinicia el kernel.'))
    return disponibles
