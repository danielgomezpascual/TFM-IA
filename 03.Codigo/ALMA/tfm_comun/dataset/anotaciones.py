"""
Lectura de las anotaciones de TUSZ (.csv y .csv_bi).

Motivo para cambiar: que cambie el formato de las anotaciones o cómo
se agrupan en crisis.
"""

from pathlib import Path

import mne
import pandas as pd


def leer_anotaciones(ruta_edf, binarias=False):
    """Lee las anotaciones de un EDF de TUSZ como tabla de pandas.

    Qué hace:
        Cada EDF tiene al lado dos ficheros con el mismo nombre:
        - .csv: anotaciones POR CANAL, con el tipo de crisis (fnsz, absz...) o bckg (fondo).
        - .csv_bi: anotaciones BINARIAS para todo el fichero: seiz (crisis) o bckg.
        Las líneas que empiezan por # son cabecera (versión, duración...) y se saltan.

    Parámetros:
        ruta_edf: ruta del .edf (se cambia la extensión para encontrar la anotación).
        binarias: False = .csv (por canal); True = .csv_bi (binarias).

    Devuelve:
        DataFrame con columnas channel, start_time, stop_time, label, confidence.
    """
    extension = '.csv_bi' if binarias else '.csv'
    return pd.read_csv(Path(ruta_edf).with_suffix(extension), comment='#')


def crisis_del_fichero(ruta_edf):
    """Lista las crisis de un fichero con su inicio, fin, tipo y duración.

    Qué hace:
        1. El INICIO y el FIN de cada crisis salen del .csv_bi (una fila 'seiz' = una crisis).
        2. El TIPO sale del .csv (por canal): dentro de cada crisis, se suma cuántos segundos ocupa
           cada tipo en cada canal (canales x segundos) y gana el que más ocupa ("tipo dominante").
        3. Si ningún canal tiene tipo dentro de esa crisis, se marca como 'sin_tipo'.

    Parámetros:
        ruta_edf: ruta del .edf.

    Devuelve:
        DataFrame con una fila por crisis: inicio_s, fin_s, tipo, duracion_s.

    Imagen: tfm_comun/imagenes/anotaciones_crisis_del_fichero.png
    """
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
    """Convierte la tabla de crisis_del_fichero() en anotaciones de MNE (una por crisis).

    Sirve para pegarlas a un Raw (raw.set_annotations(...)) y que raw.plot() pinte las crisis
    sombreadas con su tipo como texto.

    Parámetros:
        crisis: DataFrame de crisis_del_fichero() (columnas inicio_s, duracion_s, tipo).

    Devuelve:
        mne.Annotations con onset = inicio, duration = duración y description = tipo.
    """
    return mne.Annotations(onset=crisis.inicio_s.values, duration=crisis.duracion_s.values,
                           description=crisis.tipo.values)
