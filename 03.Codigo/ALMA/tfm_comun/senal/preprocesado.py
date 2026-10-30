"""
Cadena de preprocesado: en qué orden se aplican el montaje, el remuestreo y el filtrado.

Motivo para cambiar: que cambie el ORDEN o los pasos de la cadena. Cada paso está en su módulo
(montaje.py, remuestreo.py, filtrado.py); aquí solo se encadenan, para que el TFM y la demo
preprocesen exactamente igual.

    EDF referencial ──► montaje TCP (22 canales) ──► µV ──► remuestreo a 256 Hz ──► filtro 0,5-40 Hz + notch 60 Hz

Por qué remuestrear ANTES de filtrar: así el filtro se diseña siempre a 256 Hz (los mismos
coeficientes para todos los ficheros y para la demo). resample_poly ya trae su antialiasing.

Imagen: tfm_comun/imagenes/preprocesado_cadena.png (la cadena paso a paso sobre una señal sintética)
"""

import numpy as np

from tfm_comun.senal.filtrado import filtrar
from tfm_comun.senal.montaje import senal_tcp, tcp_desde_referencial
from tfm_comun.senal.remuestreo import FS_OBJETIVO, remuestrear

VOLTIOS_A_MICROVOLTIOS = 1e6   # MNE da la señal en V; se trabaja en µV (y float16 en V perdería precisión)


def preprocesar(senal_tcp_uv, fs_original, canales_validos=None):
    """Remuestrea y filtra una señal que ya está en montaje TCP y en µV (pasos 3 y 4 de la cadena).

    Qué hace:
        1. remuestrear() a 256 Hz.  2. filtrar() 0,5-40 Hz + notch 60 Hz.  3. Pasa a float32.
        4. Vuelve a poner a 0 los canales no válidos (el filtro podría dejar restos numéricos).
        Para la demo: la misma función con la señal del equipo (o FiltroContinuo, de filtrado.py,
        si la señal llega a trozos).

    Parámetros:
        senal_tcp_uv: matriz TCP (22 x muestras) en µV, a fs_original.
        fs_original: frecuencia de muestreo de la señal (Hz).
        canales_validos: 22 booleanos (de tcp_desde_referencial); None = todos válidos.

    Devuelve:
        Matriz preprocesada (22 x muestras a 256 Hz), float32, en µV.

    Imagen: tfm_comun/imagenes/preprocesado_cadena.png
    """
    senal = filtrar(remuestrear(senal_tcp_uv, fs_original), FS_OBJETIVO).astype(np.float32)
    if canales_validos is not None:
        senal[~np.asarray(canales_validos)] = 0
    return senal


def preprocesar_raw(raw):
    """La cadena completa sobre un EDF de TUSZ ya leído: montaje TCP -> µV -> 256 Hz -> filtro.

    Qué hace:
        Monta el TCP de 22 canales (senal_tcp), pasa de V a µV (x 10⁶) y llama a preprocesar().
        Es la función que usa fase_02/02_preprocesado para cada fichero.

    Parámetros:
        raw: mne.io.Raw referencial de TUSZ, leído con leer_edf_tusz() (dataset/edf.py).

    Devuelve:
        (senal, canales_validos): matriz (22 x muestras a 256 Hz) en µV y 22 booleanos.

    Imagen: tfm_comun/imagenes/preprocesado_cadena.png
    """
    tcp, validos = senal_tcp(raw)
    return preprocesar(tcp * VOLTIOS_A_MICROVOLTIOS, raw.info['sfreq'], validos), validos


def preprocesar_referencial(datos_uv, nombres, fs):
    """La cadena completa para la demo, sin MNE: electrodos del equipo -> señal lista para el modelo.

    Qué hace:
        Igual que preprocesar_raw(), pero parte de una matriz de numpy (lo que daría un equipo de EEG en
        directo) en vez de un Raw de MNE: monta el TCP (tcp_desde_referencial) y llama a preprocesar().

    Parámetros:
        datos_uv: matriz (electrodos x muestras) en µV.
        nombres: nombre limpio de cada fila ('FP1', 'A1'...).
        fs: frecuencia de muestreo del equipo (Hz).

    Devuelve:
        (senal, canales_validos): matriz (22 x muestras a 256 Hz) en µV y 22 booleanos.
    """
    tcp, validos = tcp_desde_referencial(datos_uv, nombres)
    return preprocesar(tcp, fs, validos), validos
