"""Disponibilidad del contexto: qué ventanas tienen su contexto completo y limpio, y qué ventanas hay que leer.

Motivo para cambiar: que cambien las reglas de "contexto completo y limpio".

Imagen: fase_03/img/contexto_ventanas.png

Una ventana solo tiene contexto completo si (1) el tramo empieza al menos C - 4 s antes que ella y (2) el
contexto no entra en los primeros DESCARTE_INICIO_FICHERO_S segundos del fichero (el arranque del filtro, la
misma regla que F02.03 aplica a las ventanas). La regla es la misma para las dos clases.
"""

import numpy as np
import pandas as pd

from tfm_comun.prediccion.ventanas import VENTANA_S


DESCARTE_INICIO_FICHERO_S = 10                # el contexto no entra en los primeros segundos del fichero (como F02.03)


def inicio_tramo_s(rutas):
    """Segundo del fichero en el que empieza el tramo de cada ruta (array float).

    F02.02 guarda cada tramo con su inicio en milisegundos al final del nombre:
    'train/preictal/aaaaaacz_s006_t000_preictal_000409636.npy' -> 409,636 s.
    """
    milisegundos = pd.Series(np.asarray(rutas)).astype(str).str.extract(r'_(\d+)\.npy$')[0]
    assert milisegundos.notna().all(), 'Hay rutas sin el inicio del tramo al final del nombre (_<ms>.npy).'
    return milisegundos.astype(float).to_numpy() / 1000


def con_contexto_completo(filas, k, tamano, ventana_s=VENTANA_S, descarte_s=DESCARTE_INICIO_FICHERO_S):
    """¿Tiene cada ventana su contexto de k ventanas completo y limpio? (True/False por fila).

    Dos condiciones, iguales para ⏳ y 💤:
        1. Las k - 1 ventanas anteriores caben en su tramo (la ventana empieza en la muestra muestra_inicio del tramo).
        2. El contexto no empieza en los primeros `descarte_s` segundos del fichero (arranque del filtro).
    ventana_s: segundos por ventana (tamano muestras), para pasar de muestras a segundos.
    """
    inicio = filas.muestra_inicio.to_numpy()
    cabe = inicio >= (k - 1) * tamano
    inicio_contexto_s = inicio_tramo_s(filas.ruta) + (inicio - (k - 1) * tamano) * ventana_s / tamano
    return cabe & (inicio_contexto_s >= descarte_s - 1e-6)


def inicios_necesarios(filas, k, tamano):
    """Todas las ventanas (ruta del tramo, muestra de inicio) que hacen falta para el contexto de k ventanas.

    Devuelve una tabla sin repetidos: muchas ventanas comparten anteriores (p. ej. las ⏳ seguidas de una crisis).
    """
    partes = [pd.DataFrame({'ruta': filas.ruta.to_numpy(), 'inicio': filas.muestra_inicio.to_numpy() - j * tamano})
              for j in range(k)]
    tabla = pd.concat(partes, ignore_index=True).drop_duplicates()
    assert (tabla.inicio >= 0).all(), 'Hay ventanas sin contexto completo: filtra antes con con_contexto_completo().'
    return tabla.sort_values(['ruta', 'inicio']).reset_index(drop=True)


def clave(ruta, inicio):
    """Identificador de una ventana del tramo: 'train/interictal/xxx.npy|102400'."""
    return pd.Series(ruta).astype(str).to_numpy().astype(object) + '|' + pd.Series(inicio).astype(str).to_numpy().astype(object)
