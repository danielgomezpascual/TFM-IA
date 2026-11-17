"""Corpus en memoria (F03.07): todas las ventanas del corpus sin etiquetas, fichero tras fichero, listas para sacar
secuencias seguidas.

Motivo para cambiar: que cambie cómo se carga el corpus o qué secuencias se consideran válidas.

Una secuencia válida son `largo` ventanas seguidas del MISMO fichero: si cruzara de un fichero a otro, la red
intentaría predecir el futuro de una grabación con el pasado de otra.
"""

import numpy as np
import pandas as pd

from tfm_comun.red.autosupervisado.corpus_construccion import RUTA_CORPUS, VENTANA_S


class Corpus:
    """El corpus cargado en memoria, listo para sacar secuencias seguidas de ventanas.

    Atributos:
        valores: (ventanas de todos los ficheros, 3.520) float16, fichero tras fichero.
        paciente: paciente de cada ventana (array de str).  inicio_fichero: posición de la 1.ª ventana de cada fichero.
        ficheros: tabla de corpus.csv con los ficheros cargados.
    """

    def __init__(self, ruta=RUTA_CORPUS, pacientes=None):
        tabla = pd.read_csv(ruta / 'corpus.csv')
        tabla = tabla[(tabla.estado == 'hecho') & (tabla.ventanas > 0)]
        if pacientes is not None:
            tabla = tabla[tabla.patient.isin(pacientes)]
        self.ficheros = tabla.reset_index(drop=True)
        total = int(self.ficheros.ventanas.sum())
        primero = np.load(ruta / 'corpus' / f'{self.ficheros.file.iloc[0]}.npy', mmap_mode='r')
        self.valores = np.empty((total, primero.shape[1]), dtype=np.float16)
        self.paciente = np.empty(total, dtype=object)
        self.inicio_fichero = np.zeros(len(self.ficheros), dtype=np.int64)
        a = 0
        for i, f in enumerate(self.ficheros.itertuples()):
            v = np.load(ruta / 'corpus' / f'{f.file}.npy')
            self.valores[a:a + len(v)] = v
            self.paciente[a:a + len(v)] = f.patient
            self.inicio_fichero[i] = a
            a += len(v)

    def inicios_validos(self, largo):
        """Posiciones donde empieza una secuencia de `largo` ventanas seguidas DEL MISMO fichero."""
        partes = [np.arange(a, a + n - largo + 1) for a, n in zip(self.inicio_fichero, self.ficheros.ventanas) if n >= largo]
        return np.concatenate(partes) if partes else np.empty(0, dtype=np.int64)

    @property
    def horas(self):
        return len(self.valores) * VENTANA_S / 3600
