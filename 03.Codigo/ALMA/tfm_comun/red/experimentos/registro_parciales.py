"""Registro de entrenamientos parciales: cada entrenamiento suelto (una semilla en una ronda) en cuanto acaba.

Motivo para cambiar: que cambie qué se guarda de un entrenamiento suelto o dónde.

Para qué: una variante son varias semillas x varias rondas (p. ej. 3 x 1 en 🟡 dev, o 3 x 5 con pliegues). Si el
notebook se corta en la tercera semilla, al volver no se repiten las dos primeras: se leen de aquí.

En la carpeta:  <huella>_<semilla>_<ronda>.npz   puntuaciones de esa semilla en las filas de validación (+ sus id)
                <huella>_<semilla>_<ronda>.csv   su curva de aprendizaje
"""

import numpy as np
import pandas as pd


class RegistroParciales:
    """Los entrenamientos sueltos de una carpeta (se crea al guardar el primero)."""

    def __init__(self, carpeta):
        self.carpeta = carpeta

    def _rutas(self, h, semilla, ronda):
        nombre = f"{h}_{semilla}_{''.join(c if c.isalnum() else '_' for c in str(ronda))}"
        return self.carpeta / f'{nombre}.npz', self.carpeta / f'{nombre}.csv'

    def guardar(self, h, semilla, ronda, puntuaciones, ids, historial):
        """Apunta un entrenamiento suelto en cuanto acaba: sus puntuaciones (en las filas `ids`) y su historial."""
        ruta_npz, ruta_csv = self._rutas(h, semilla, ronda)
        ruta_npz.parent.mkdir(parents=True, exist_ok=True)
        historial.to_csv(ruta_csv, index=False)
        np.savez_compressed(ruta_npz, puntuaciones=np.asarray(puntuaciones, dtype=np.float32), ids=np.asarray(ids).astype(str))

    def buscar(self, h, semilla, ronda, ids):
        """(puntuaciones, historial) de un entrenamiento suelto ya hecho con estas mismas filas, o None."""
        ruta_npz, ruta_csv = self._rutas(h, semilla, ronda)
        if not (ruta_npz.exists() and ruta_csv.exists()):
            return None
        guardado = np.load(ruta_npz, allow_pickle=False)
        if len(guardado['ids']) != len(ids) or not (guardado['ids'] == np.asarray(ids).astype(str)).all():
            return None
        return guardado['puntuaciones'], pd.read_csv(ruta_csv)

    def borrar(self, h):
        """Borra los entrenamientos sueltos de una huella (al reentrenar desde cero)."""
        if self.carpeta.exists():
            for ruta in self.carpeta.glob(f'{h}_*'):
                ruta.unlink()
