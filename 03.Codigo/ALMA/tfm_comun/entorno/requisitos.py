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
import subprocess
import sys

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
    print(f'Instalando: {" ".join(nombres_pip)} (puede tardar un par de minutos)...')
    resultado = subprocess.run([sys.executable, '-m', 'pip', 'install', *nombres_pip], capture_output=True, text=True)
    importlib.invalidate_caches()
    if resultado.returncode != 0:
        print(resultado.stderr[-2000:])


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
    if not faltan:
        print('Todo lo necesario está instalado.' + ('' if all(antes.values()) else
              ' Si una recién instalada diera error al importarla, reinicia el kernel.'))
    return disponibles
