"""
Catálogo de pacientes y ficheros de TUSZ, para elegir qué mirar (lo usa el visor de EEG).

Motivo para cambiar: que cambie de dónde sale la lista de pacientes y ficheros.
No lee ningún EDF: sale del inventario de F01.01 (outputs/fase_01/01_inventario/edfs.csv),
que tiene una fila por EDF con su partición, paciente, montaje, duración y crisis.
"""

import pandas as pd

from tfm_comun.entorno.rutas import RUTA_OUTPUTS_FASE_01

RUTA_INVENTARIO_EDFS = RUTA_OUTPUTS_FASE_01 / '01_inventario' / 'edfs.csv'


def leer_inventario():
    """Lee el inventario de EDF de F01.01 (una fila por fichero).

    Columnas que se usan: split, patient, session, token, montage, file, edf_dur_s,
    n_seiz_events_bi (crisis del .csv_bi) y labels (tipos del fichero separados por ';').
    Si no existe, para con un error que dice qué notebook hay que ejecutar antes.
    """
    assert RUTA_INVENTARIO_EDFS.exists(), (f'No está {RUTA_INVENTARIO_EDFS}. '
                                           'Ejecuta antes F01.01 (fase_01/01_inventario_tusz.ipynb).')
    inventario = pd.read_csv(RUTA_INVENTARIO_EDFS)
    inventario['labels'] = inventario['labels'].fillna('')
    return inventario


def resumen_pacientes(inventario, split):
    """Una fila por paciente de una partición: cuántos ficheros, horas, crisis y de qué tipos.

    Parámetros:
        inventario: tabla de leer_inventario().
        split: 'train', 'dev' o 'eval'.

    Devuelve:
        DataFrame con índice paciente y columnas n_ficheros, horas, n_crisis y tipos (lista ordenada),
        ordenado de más a menos crisis.
    """
    del_split = inventario[inventario.split == split]
    resumen = del_split.groupby('patient').agg(n_ficheros=('file', 'size'),
                                                horas=('edf_dur_s', lambda s: s.sum() / 3600),
                                                n_crisis=('n_seiz_events_bi', 'sum'),
                                                tipos=('labels', _tipos_unicos))
    return resumen.sort_values(['n_crisis', 'horas'], ascending=False)


def ficheros_paciente(inventario, paciente):
    """Los EDF de un paciente, en orden de sesión y token, con su duración y sus crisis."""
    del_paciente = inventario[inventario.patient == paciente]
    return del_paciente.sort_values('file').reset_index(drop=True)


def _tipos_unicos(etiquetas):
    """De varias celdas 'fnsz;gnsz', 'fnsz', '' a la lista ['fnsz', 'gnsz'] (sin repetir, ordenada)."""
    return sorted({t for celda in etiquetas for t in celda.split(';') if t})
