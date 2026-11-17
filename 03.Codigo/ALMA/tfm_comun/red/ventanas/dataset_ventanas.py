"""Dataset de ventanas (VentanasEEG): las ventanas de 4 s de una vista, leídas una a una del paquete de su partición.

Motivo para cambiar: que cambie cómo se lee una ventana del disco.

Qué es un Dataset de PyTorch
============================
Una clase con dos métodos: __len__ (cuántos ejemplos hay) y __getitem__(i) (el ejemplo n.º i). No sabe nada de
lotes ni de barajar: eso lo hace el DataLoader (red/ventanas/cargador.py). Separarlo así permite cambiar cómo se
leen los datos sin tocar cómo se reparten en lotes, y al revés.

Imagen: fase_03/img/dataset_y_dataloader.png (qué hace el Dataset y qué hace el DataLoader)
"""

import numpy as np
from torch.utils.data import Dataset


class VentanasEEG(Dataset):
    """Las ventanas de una vista, leídas del paquete de su partición.

    Parámetros:
        ruta_paquete: el .npy del paquete (ventanas x canales x muestras).
        posiciones: filas del paquete que forman la vista (p. ej. las de en_entrenamiento = True).
        etiquetas: etiqueta de cada una de esas filas (1 = ⏳ preictal, 0 = 💤 interictal).

    El paquete se abre con mmap la primera vez que se pide una ventana, y no se guarda al copiar el Dataset
    (__getstate__): en Windows, cada proceso que ayuda a cargar (n_trabajadores > 0) recibe una copia y abre
    el paquete por su cuenta, sin copiar gigas de datos entre procesos.
    """

    def __init__(self, ruta_paquete, posiciones, etiquetas):
        self.ruta_paquete = str(ruta_paquete)
        self.posiciones = np.asarray(posiciones, dtype=np.int64)
        self.etiquetas = np.asarray(etiquetas, dtype=np.float32)
        assert len(self.posiciones) == len(self.etiquetas), 'Hay que dar una etiqueta por posición.'
        self._datos = None

    @property
    def datos(self):
        if self._datos is None:
            self._datos = np.load(self.ruta_paquete, mmap_mode='r')
        return self._datos

    def __len__(self):
        return len(self.posiciones)

    def __getitem__(self, i):
        return np.asarray(self.datos[self.posiciones[i]]), self.etiquetas[i]

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado['_datos'] = None
        return estado
