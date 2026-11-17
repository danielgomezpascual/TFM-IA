"""Registro de experimentos: cada red entrenada (su configuración, su nota y sus puntuaciones), para no volver a
entrenarla y poder ver cómo mejora la red a lo largo del TFM.

Motivo para cambiar: que cambie qué se apunta de cada experimento terminado o dónde se guarda.

Imagen: fase_03/img/escalera_mejoras.png (la escalera de mejoras y el registro)

Cómo funciona
=============
Cada experimento es una CONFIGURACIÓN (arquitectura + entrenamiento + cómo se eligen las filas). De ella sale una
HUELLA (red/experimentos/huella.py). En la carpeta del registro:

    registro.csv                          una fila por experimento: huella, nombre, fecha, notas en dev, configuración
    puntuaciones/<huella>.npz             la probabilidad de cada semilla en cada ventana de validación (+ sus id)
    historial/<huella>.csv                las curvas de aprendizaje de cada semilla
    parciales/<huella>_<semilla>_<ronda>  lo de cada entrenamiento suelto, en cuanto acaba (RegistroParciales)

Antes de entrenar, el notebook mira si la huella ya está y si las ventanas de validación son las mismas: si es así,
lee las puntuaciones y no entrena. Así, añadir un experimento nuevo a la lista solo entrena ese.
"""

import numpy as np
import pandas as pd

from tfm_comun.red.experimentos.registro_parciales import RegistroParciales


class Registro:
    """El registro de experimentos de una carpeta (se crea si no existe).

    Atributo parciales: RegistroParciales de la subcarpeta 'parciales' (los entrenamientos sueltos).
    """

    def __init__(self, carpeta):
        self.carpeta = carpeta
        (carpeta / 'puntuaciones').mkdir(parents=True, exist_ok=True)
        (carpeta / 'historial').mkdir(parents=True, exist_ok=True)
        self.ruta = carpeta / 'registro.csv'
        self.parciales = RegistroParciales(carpeta / 'parciales')

    def tabla(self):
        """Todo el registro (una fila por experimento), o una tabla vacía."""
        return pd.read_csv(self.ruta) if self.ruta.exists() else pd.DataFrame(columns=['huella'])

    def buscar(self, h, ids_validacion):
        """(puntuaciones, historial) de un experimento ya hecho con estas mismas ventanas de validación, o None.

        puntuaciones: array (semillas, ventanas de validación) en el orden de ids_validacion.
        """
        ruta_npz = self.carpeta / 'puntuaciones' / f'{h}.npz'
        ruta_hist = self.carpeta / 'historial' / f'{h}.csv'
        if not (ruta_npz.exists() and ruta_hist.exists()):
            return None
        guardado = np.load(ruta_npz, allow_pickle=False)
        if len(guardado['ids']) != len(ids_validacion) or not (guardado['ids'] == np.asarray(ids_validacion).astype(str)).all():
            return None                                   # otras ventanas (ha cambiado la Fase 2 o F03.01): hay que repetirlo
        return guardado['puntuaciones'], pd.read_csv(ruta_hist)

    def guardar(self, h, fila, puntuaciones, ids_validacion, historial):
        """Apunta un experimento (sustituye la fila si la huella ya estaba)."""
        np.savez_compressed(self.carpeta / 'puntuaciones' / f'{h}.npz', puntuaciones=np.asarray(puntuaciones, dtype=np.float32),
                            ids=np.asarray(ids_validacion).astype(str))
        historial.to_csv(self.carpeta / 'historial' / f'{h}.csv', index=False)
        tabla = self.tabla()
        tabla = tabla[tabla.huella != h] if len(tabla) else tabla
        nueva = pd.DataFrame([{'huella': h, **fila}])
        (nueva if not len(tabla) else pd.concat([tabla, nueva], ignore_index=True)).to_csv(self.ruta, index=False)
