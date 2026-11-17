"""Requisitos de cada notebook: qué librerías necesita, si están instaladas y, si no, instalarlas.

Motivo para cambiar: que un notebook necesite librerías nuevas o cambie cómo se enseña la comprobación.
Cómo se detecta la GPU: entorno/gpu_nvidia.py. Cómo se instala con pip: entorno/instalacion.py.
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

import pandas as pd
from IPython.display import Markdown, display

from tfm_comun.entorno.instalacion import instalada, instalar, reinstalar_torch_gpu, version_instalada


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


REQUISITOS_FASE_03_LINEA_BASE = {
    'numpy': ('numpy', 'cálculo con matrices', True),
    'pandas': ('pandas', 'tablas', True),
    'matplotlib': ('matplotlib', 'figuras', True),
    'scipy': ('scipy', 'espectro de cada ventana (Welch)', True),
    'sklearn': ('scikit-learn', 'regresión logística, AUROC y curvas ROC', True),
    'xgboost': ('xgboost', 'XGBoost: árboles de decisión encadenados', True),
    'joblib': ('joblib', 'guardar los modelos entrenados', True),
    'tqdm': ('tqdm', 'barra de progreso al calcular las características', True),
}


# F03.03 (contexto y normalización por paciente) usa las mismas librerías que la línea base
REQUISITOS_FASE_03_CONTEXTO = REQUISITOS_FASE_03_LINEA_BASE


# F03.04 (la red neuronal): las de la línea base (espectro, AUC, tablas) + PyTorch
REQUISITOS_FASE_03_RED = {
    **REQUISITOS_FASE_03_LINEA_BASE,
    'torch': ('torch', 'la red neuronal: capas, entrenamiento en la GPU (PyTorch)', True),
}


# herramientas/visor_eeg.ipynb: ver el EEG de un paciente con las crisis marcadas
REQUISITOS_VISOR = {
    'numpy': ('numpy', 'cálculo con matrices', True),
    'pandas': ('pandas', 'tablas (inventario de F01.01 y anotaciones)', True),
    'matplotlib': ('matplotlib', 'dibujar el EEG', True),
    'scipy': ('scipy', 'filtro paso banda y notch', True),
    'mne': ('mne', 'leer el EDF y montaje TCP', True),
    'ipywidgets': ('ipywidgets', 'los controles del visor (desplegables, deslizador, botones)', True),
    'pyvista': ('pyvista', 'el cerebro 3D del botón 🧠 (sin él, solo la cabeza)', False),
}


def comprobar_requisitos(requisitos, instalar_faltan=True):
    """Comprueba que están instaladas las librerías del notebook, instala las que falten y lo enseña en una tabla.

    Parámetros:
        requisitos: diccionario como REQUISITOS_03V.
        instalar_faltan: si es True (por defecto), instala con pip lo que falte.
    Qué hace:
        1. Mira si cada librería se puede importar.
        2. Si falta alguna y instalar_faltan=True, la instala con pip (en el Python del notebook) y vuelve a mirar.
        3. Enseña una tabla: ✅ ya estaba (con versión), 🆕 recién instalada, ❌ falta (obligatoria),
           🔴 falta (opcional).
        4. Si sigue faltando alguna obligatoria, para con un error. Si solo faltan opcionales, avisa y sigue.
    Devuelve: {nombre para importar: True/False}, para saber luego qué partes se pueden ejecutar.
    """
    antes = {modulo: instalada(modulo) for modulo in requisitos}
    if instalar_faltan and not all(antes.values()):
        instalar([requisitos[m][0] for m, ok in antes.items() if not ok])
    reinstalado = instalar_faltan and 'torch' in requisitos and antes.get('torch') and reinstalar_torch_gpu()

    filas, disponibles = [], {}
    for modulo, (nombre_pip, para_que, obligatoria) in requisitos.items():
        disponibles[modulo] = instalada(modulo)
        if disponibles[modulo]:
            estado = '✅' if antes[modulo] else '🆕 recién instalada'
        else:
            estado = '❌ falta' if obligatoria else '🔴 falta (opcional)'
        filas.append({'librería': nombre_pip, 'estado': estado, 'versión': version_instalada(nombre_pip) or '-',
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
