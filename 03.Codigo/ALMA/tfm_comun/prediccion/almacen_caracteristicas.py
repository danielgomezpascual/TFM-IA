"""
Almacén de características: guardar en disco las de cada paquete y reutilizarlas si no ha cambiado nada.

Motivo para cambiar: que cambie dónde o cómo se guardan las características.
Qué se mide está en caracteristicas.py; aquí solo se guarda y se lee.

    RUTA_CARACTERISTICAS/<conjunto>/<split>.npy        (ventanas x características), float32 (o float16 si llegan así)
    RUTA_CARACTERISTICAS/<conjunto>/<split>_ids.csv    id_ventana de cada fila, en el mismo orden

Calcularlas lleva unos minutos (hay que leer todo el paquete); guardarlas hace que la segunda vez tarde segundos.
Si el paquete de F03.01 cambia (otras ventanas), los id_ventana ya no coinciden y se vuelven a calcular.
"""

import numpy as np
import pandas as pd

from tfm_comun.entorno.rutas import RUTA_CARACTERISTICAS


def rutas_caracteristicas(conjunto, split):
    """(ruta del .npy, ruta del .csv de ids) de las características de un conjunto y una partición."""
    carpeta = RUTA_CARACTERISTICAS / conjunto
    return carpeta / f'{split}.npy', carpeta / f'{split}_ids.csv'


def caracteristicas_al_dia(conjunto, split, ids, n_caracteristicas):
    """¿Ya hay características guardadas para EXACTAMENTE estas ventanas, en este orden y con este número de columnas?"""
    ruta_npy, ruta_ids = rutas_caracteristicas(conjunto, split)
    if not (ruta_npy.exists() and ruta_ids.exists()):
        return False
    guardados = pd.read_csv(ruta_ids).id_ventana.to_numpy()
    forma = np.load(ruta_npy, mmap_mode='r').shape
    return len(guardados) == len(ids) and (guardados == np.asarray(ids)).all() and forma == (len(ids), n_caracteristicas)


def guardar_caracteristicas(conjunto, split, ids, caracteristicas):
    """Guarda las características y sus id_ventana (primero en .tmp y luego renombra: nunca queda a medias)."""
    ruta_npy, ruta_ids = rutas_caracteristicas(conjunto, split)
    ruta_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy, tmp_ids = ruta_npy.with_name(ruta_npy.stem + '_tmp.npy'), ruta_ids.with_name(ruta_ids.stem + '_tmp.csv')
    caracteristicas = np.asarray(caracteristicas)
    tipo = np.float16 if caracteristicas.dtype == np.float16 else np.float32       # float16 se queda (F03.06: la mitad de disco)
    np.save(tmp_npy, caracteristicas.astype(tipo, copy=False))
    pd.DataFrame({'id_ventana': ids}).to_csv(tmp_ids, index=False)
    tmp_npy.replace(ruta_npy)
    tmp_ids.replace(ruta_ids)


def cargar_caracteristicas(conjunto, split):
    """(matriz de características, id_ventana de cada fila)."""
    ruta_npy, ruta_ids = rutas_caracteristicas(conjunto, split)
    assert ruta_npy.exists(), f'No encuentro {ruta_npy}. Ejecuta antes F03.02 (fase_03/02_linea_base.ipynb).'
    return np.load(ruta_npy), pd.read_csv(ruta_ids).id_ventana.to_numpy()
