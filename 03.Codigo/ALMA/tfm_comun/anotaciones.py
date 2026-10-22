"""
Lectura de las anotaciones de TUSZ (.csv y .csv_bi).

Motivo para cambiar: que cambie el formato de las anotaciones o cómo
se agrupan en crisis.
"""

from pathlib import Path

import mne
import pandas as pd


def leer_anotaciones(ruta_edf, binarias=False):
    """Tabla de anotaciones del fichero: .csv (por canal, con tipo de crisis)
    o .csv_bi (binarias seiz/bckg, para todo el fichero). Las líneas # son cabecera."""
    extension = '.csv_bi' if binarias else '.csv'
    return pd.read_csv(Path(ruta_edf).with_suffix(extension), comment='#')


def crisis_del_fichero(ruta_edf):
    """Una fila por crisis: inicio y fin del .csv_bi, y tipo dominante del .csv
    (el que más canales×segundos ocupa dentro de esa crisis)."""
    binarias = leer_anotaciones(ruta_edf, binarias=True)
    por_canal = leer_anotaciones(ruta_edf)
    por_canal = por_canal[por_canal.label != 'bckg']
    filas = []
    for _, c in binarias[binarias.label == 'seiz'].iterrows():
        solape = (por_canal.stop_time.clip(upper=c.stop_time)
                  - por_canal.start_time.clip(lower=c.start_time)).clip(lower=0)
        tipos = solape.groupby(por_canal.label).sum()
        tipo = tipos.idxmax() if tipos.sum() > 0 else 'sin_tipo'
        filas.append({'inicio_s': c.start_time, 'fin_s': c.stop_time, 'tipo': tipo})
    crisis = pd.DataFrame(filas, columns=['inicio_s', 'fin_s', 'tipo'])
    crisis['duracion_s'] = crisis.fin_s - crisis.inicio_s
    return crisis


def a_anotaciones_mne(crisis):
    """Convierte la tabla de crisis_del_fichero() en mne.Annotations (una por crisis)."""
    return mne.Annotations(onset=crisis.inicio_s.values, duration=crisis.duracion_s.values,
                           description=crisis.tipo.values)
