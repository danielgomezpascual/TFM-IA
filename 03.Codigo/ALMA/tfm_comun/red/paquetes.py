"""
Paquetes: las ventanas de un conjunto y una partición, leídas UNA vez y guardadas juntas, listas para la red.

Motivo para cambiar: que cambie cómo se guardan en disco las ventanas que entran en la red.

Imagen: fase_03/img/del_indice_al_lote.png (del índice de ventanas al lote que entra en la red)

Por qué un paquete
==================
La señal preprocesada está en miles de .npy (uno por tramo, F02.02) y cada ventana es un trozo de
uno de ellos (ruta + muestra_inicio). Entrenar es leer las mismas ventanas muchas veces (una por época):
abrir miles de ficheros en cada época es lento. Un paquete las junta en UNA matriz, en el orden del índice:

    RUTA_PAQUETES/<conjunto>/<split>.npy          (ventanas x 22 canales x 1.024 muestras), float16, µV, sin normalizar
    RUTA_PAQUETES/<conjunto>/<split>_indice.csv   una fila por ventana, en el mismo orden: id, paciente, clase,
                                                  etiqueta... y a qué vistas pertenece (en_entrenamiento, en_evaluacion...)

Se guarda SIN normalizar (como en F02.02): la normalización se hace al vuelo, en cada lote (red/datos.py), para
poder cambiarla sin rehacer los paquetes. El .npy se abre con mmap: no hace falta que quepa en la memoria.
"""

import os

import numpy as np
import pandas as pd

from tfm_comun.entorno.rutas import RUTA_PAQUETES, RUTA_SENALES
from tfm_comun.prediccion.ventanas import muestras_por_ventana, ventanas_de_senal
from tfm_comun.senal.senales import TIPO_DATO, cargar_tramo


def rutas_paquete(conjunto, split):
    """(ruta del .npy, ruta del índice .csv) del paquete de un conjunto y una partición."""
    carpeta = RUTA_PAQUETES / conjunto
    return carpeta / f'{split}.npy', carpeta / f'{split}_indice.csv'


def paquete_al_dia(conjunto, split, indice, n_canales=22, tamano=None):
    """¿Ya existe un paquete con EXACTAMENTE estas ventanas, en este orden y con esta forma?

    Compara el id_ventana del índice guardado con el nuevo y la forma del .npy. Si cambia algo en la Fase 2
    (otras ventanas, otro equilibrado...), deja de estar al día y se vuelve a crear.
    """
    ruta_npy, ruta_csv = rutas_paquete(conjunto, split)
    if not (ruta_npy.exists() and ruta_csv.exists()):
        return False
    guardado = pd.read_csv(ruta_csv, usecols=['id_ventana']).id_ventana
    forma = np.load(ruta_npy, mmap_mode='r').shape
    return (len(guardado) == len(indice) and (guardado.to_numpy() == indice.id_ventana.to_numpy()).all()
            and forma == (len(indice), n_canales, tamano or muestras_por_ventana()))


def crear_paquete(conjunto, split, indice, n_canales=22, tamano=None, progreso=None):
    """Lee las ventanas del índice desde los .npy de F02.02 y las guarda juntas en el paquete.

    Qué hace:
        1. Crea el .npy vacío del tamaño final (ventanas x canales x muestras) directamente en disco
           (open_memmap): no hace falta tener todo en memoria.
        2. Recorre los .npy de F02.02 de uno en uno (agrupando las ventanas por ruta) y copia cada ventana a su fila.
           Cada tramo se abre una sola vez y con mmap: solo se lee lo que se copia.
        3. Escribe en ficheros .tmp y al final los renombra: si se corta, nunca queda un paquete a medias.

    Parámetros:
        indice: tabla de indice_por_particion() para este split (una fila por ventana, sin repetir).
        tamano: muestras por ventana (por defecto las de prediccion/ventanas.py: 1.024).
        progreso: función tipo tqdm para envolver el bucle (opcional).

    Devuelve:
        Diccionario con ventanas, ficheros .npy leídos y GB del paquete.
    """
    tamano = tamano or muestras_por_ventana()
    ruta_npy, ruta_csv = rutas_paquete(conjunto, split)
    ruta_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy, tmp_csv = ruta_npy.with_name(ruta_npy.name + '.tmp'), ruta_csv.with_name(ruta_csv.name + '.tmp')
    indice = indice.reset_index(drop=True)

    salida = np.lib.format.open_memmap(tmp_npy, mode='w+', dtype=TIPO_DATO, shape=(len(indice), n_canales, tamano))
    grupos = indice.groupby('ruta', sort=False).indices          # {ruta: posiciones en el índice}
    recorrido = progreso(grupos.items(), total=len(grupos)) if progreso else grupos.items()
    for ruta, posiciones in recorrido:
        senal = cargar_tramo(RUTA_SENALES / ruta)               # mmap: no lee el tramo entero
        inicios = indice.muestra_inicio.to_numpy()[posiciones]
        assert inicios.max() + tamano <= senal.shape[-1], f'Ventana fuera del tramo en {ruta}.'
        salida[posiciones] = ventanas_de_senal(senal, tamano, inicios)
        del senal
    salida.flush()
    del salida                                                  # cierra el fichero (Windows no deja renombrarlo abierto)
    indice.to_csv(tmp_csv, index=False)
    os.replace(tmp_npy, ruta_npy)
    os.replace(tmp_csv, ruta_csv)
    return {'ventanas': len(indice), 'ficheros .npy leídos': len(grupos), 'GB': ruta_npy.stat().st_size / 1e9}


def cargar_paquete(conjunto, split):
    """Abre un paquete: (matriz mmap ventanas x canales x muestras en float16, tabla índice)."""
    ruta_npy, ruta_csv = rutas_paquete(conjunto, split)
    assert ruta_npy.exists(), f'No encuentro {ruta_npy}. Ejecuta antes F03.01 (fase_03/01_datos_red.ipynb) con CONJUNTO = {conjunto!r}.'
    return np.load(ruta_npy, mmap_mode='r'), pd.read_csv(ruta_csv)
