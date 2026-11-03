"""
Vistas: qué ventanas de un conjunto se usan para entrenar, para elegir, para la nota final y en cada control.

Motivo para cambiar: que cambie qué ventanas forman cada vista o qué conjuntos de control hay.
Solo trabaja con tablas (no lee la señal).

Qué es una vista
================
F02.04 (`04_particiones_y_balanceo`) deja un conjunto por modelo (p. ej. conjuntos/general.csv) y F02.05
(`05_auditoria_atajos`) deja sus conjuntos de control (conjuntos_control/general_*.csv). Una VISTA es el trozo de esas tablas que se usa
para una cosa concreta:

    entrenamiento                   🔵 train con entrenamiento_equilibrado = True: lo que ve la red al aprender
    validacion                      🟡 dev entero, con su desbalance real: elegir época, hiperparámetros y umbral
    evaluacion                      🟢 eval entero, con su desbalance real: la nota final (se mira al final)
    control_posicion                🟢 eval del control emparejado por posición (H4 de F02.05)
    control_misma_sesion            🟢 eval del control con 💤 de la misma sesión (H5 de F02.05)
    control_misma_sesion_estricta   el anterior, solo con 💤 de ficheros anteriores a la primera crisis

En los conjuntos con validación cruzada por paciente (los que traen la columna `pliegue`, p. ej. grupo_ausencia)
no hay validacion: hay validacion_cruzada (train + dev enteros, con su pliegue), y el entrenamiento es train + dev.
"""

import pandas as pd

from tfm_comun.dataset.particiones import SPLITS
from tfm_comun.entorno.rutas import RUTA_OUTPUTS_FASE_02

RUTA_CONJUNTOS = RUTA_OUTPUTS_FASE_02 / '04_particiones_y_balanceo' / 'conjuntos'
RUTA_CONTROLES = RUTA_OUTPUTS_FASE_02 / '05_auditoria_atajos' / 'conjuntos_control'

# Qué es cada vista y para qué se usa (en este orden salen en todas las tablas)
DESCRIPCION_VISTAS = {
    'entrenamiento': 'Entrenar la red (equilibrado: mitad y mitad en cada paciente)',
    'validacion': 'Elegir época, hiperparámetros y umbral (desbalance real)',
    'validacion_cruzada': 'Validación cruzada por paciente (train + dev enteros, con su pliegue)',
    'evaluacion': 'Nota final (desbalance real). Solo se mira al final',
    'control_posicion': 'Control H4: 💤 de los mismos minutos del fichero que las ⏳',
    'control_misma_sesion': 'Control H5: 💤 de ficheros sin crisis de la misma sesión',
    'control_misma_sesion_estricta': 'Control H5 estricto: solo 💤 de ficheros anteriores a la 1.ª crisis',
}

# Ficheros de control de F02.05: sufijo del fichero -> vista
SUFIJOS_CONTROL = {'emparejado_posicion': 'control_posicion', 'misma_sesion': 'control_misma_sesion'}

# Columnas que se quedan de cada ventana (las que necesita la Fase 3 para entrenar y evaluar)
COLUMNAS = ['id_ventana', 'split', 'patient', 'file', 'clase', 'etiqueta', 'tipo', 'grupo', 'crisis_id',
            'ruta', 'muestra_inicio', 'segundos_hasta_crisis', 'pliegue']


def leer_conjunto(nombre):
    """Lee conjuntos/<nombre>.csv de F02.04. Para con un error claro si no existe."""
    ruta = RUTA_CONJUNTOS / f'{nombre}.csv'
    assert ruta.exists(), f'No encuentro {ruta}. Ejecuta antes F02.04 (fase_02/04_particiones_y_balanceo.ipynb).'
    return pd.read_csv(ruta)


def leer_controles(nombre):
    """Los conjuntos de control de F02.05 para el conjunto `nombre`, como {vista: tabla}.

    Busca conjuntos_control/<nombre>_<sufijo>.csv. Si un control no existe (F02.05 no lo creó porque no hacía
    falta o no había datos), simplemente no aparece. Del control de misma sesión sale también la variante
    estricta: las ⏳ preictales de siempre y solo las 💤 con antes_de_la_primera_crisis = True.
    """
    controles = {}
    for sufijo, vista in SUFIJOS_CONTROL.items():
        ruta = RUTA_CONTROLES / f'{nombre}_{sufijo}.csv'
        if ruta.exists():
            controles[vista] = pd.read_csv(ruta)
    if 'control_misma_sesion' in controles and 'antes_de_la_primera_crisis' in controles['control_misma_sesion']:
        c = controles['control_misma_sesion']
        estricta = (c.clase == 'preictal') | c.antes_de_la_primera_crisis.astype(bool)
        controles['control_misma_sesion_estricta'] = c[estricta].copy()
    return controles


def tiene_validacion_cruzada(conjunto):
    """True si el conjunto trae pliegues por paciente (lo decide F02.04 con GRUPOS_VALIDACION_CRUZADA)."""
    return 'pliegue' in conjunto.columns and conjunto.pliegue.notna().any()


def armar_vistas(conjunto, controles, splits_control=('eval',)):
    """Las vistas de un conjunto, como {vista: tabla}, en el orden de DESCRIPCION_VISTAS.

    Parámetros:
        conjunto: tabla de leer_conjunto().
        controles: {vista: tabla} de leer_controles().
        splits_control: de qué particiones se quedan los controles. Por defecto solo 🟢 eval: los controles
            sirven para comprobar la nota final (F02.05: "evaluar la red también en los controles").

    Devuelve:
        {vista: tabla con las COLUMNAS que tenga}. Cada fila es una ventana.
    """
    columnas = [c for c in COLUMNAS if c in conjunto.columns]
    vistas = {}
    if tiene_validacion_cruzada(conjunto):
        entrena = conjunto.split.isin(['train', 'dev'])
        vistas['entrenamiento'] = conjunto[entrena & conjunto.entrenamiento_equilibrado]
        vistas['validacion_cruzada'] = conjunto[entrena & conjunto.pliegue.notna()]
    else:
        vistas['entrenamiento'] = conjunto[(conjunto.split == 'train') & conjunto.entrenamiento_equilibrado]
        vistas['validacion'] = conjunto[conjunto.split == 'dev']
    vistas['evaluacion'] = conjunto[conjunto.split == 'eval']
    for vista, tabla in controles.items():
        vistas[vista] = tabla[tabla.split.isin(splits_control)]
    orden = [v for v in DESCRIPCION_VISTAS if v in vistas]
    return {v: vistas[v][[c for c in columnas if c in vistas[v].columns]].reset_index(drop=True) for v in orden}


def tabla_vistas(vistas):
    """Resumen de cada vista: partición, pacientes, crisis, ventanas de cada clase y desbalance."""
    filas = {}
    for vista, t in vistas.items():
        n_pre, n_inter = int((t.clase == 'preictal').sum()), int((t.clase == 'interictal').sum())
        filas[vista] = {'partición': ' + '.join(s for s in SPLITS if s in set(t.split)),
                        'para qué': DESCRIPCION_VISTAS[vista],
                        'pacientes': t.patient.nunique(), 'crisis': t.crisis_id.nunique(),
                        '⏳ preictal': n_pre, '💤 interictal': n_inter,
                        '💤 por cada ⏳': round(n_inter / n_pre, 2) if n_pre else float('nan'),
                        'ventanas': len(t)}
    tabla = pd.DataFrame(filas).T
    tabla.index.name = 'vista'
    return tabla


def indice_por_particion(vistas):
    """Une las vistas por partición: una tabla por split con cada ventana UNA sola vez y a qué vistas pertenece.

    Qué hace:
        Una misma ventana puede estar en varias vistas (p. ej. una ⏳ de eval está en evaluacion y en los tres
        controles). Para no leerla ni guardarla varias veces, se juntan todas las ventanas de cada partición sin
        repetir (por id_ventana) y se añade una columna verdadero/falso por vista: en_<vista>.
        Se ordenan por paciente, fichero y muestra: así, al leer la señal, cada .npy se recorre de principio a fin.

    Devuelve:
        {split: tabla}, solo con las particiones que aparecen en alguna vista.
    """
    todas = pd.concat([t.assign(_vista=v) for v, t in vistas.items()], ignore_index=True)
    indices = {}
    for split in [s for s in SPLITS if s in set(todas.split)]:
        parte = todas[todas.split == split]
        unicas = parte.drop(columns='_vista').drop_duplicates('id_ventana').set_index('id_ventana')
        for vista in vistas:
            unicas[f'en_{vista}'] = unicas.index.isin(parte.loc[parte._vista == vista, 'id_ventana'])
        indices[split] = unicas.reset_index().sort_values(['patient', 'file', 'ruta', 'muestra_inicio']).reset_index(drop=True)
    return indices
