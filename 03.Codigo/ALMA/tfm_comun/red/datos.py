"""
Datos de la red: cómo le llegan las ventanas a PyTorch (Dataset), cómo se preparan en lotes y cómo se sirven (DataLoader).

Motivo para cambiar: que cambie cómo se leen, se preparan o se reparten en lotes las ventanas.

Imagen: fase_03/img/del_indice_al_lote.png (del índice de ventanas al lote que entra en la red)

Las tres piezas de PyTorch
==========================
- Dataset (VentanasEEG): sabe cuántas ventanas hay y devuelve UNA cuando se le pide la n.º i: la matriz
  (22 x 1.024) en float16 y µV, sacada del paquete (red/paquetes.py), y su etiqueta (1 = ⏳ preictal, 0 = 💤 interictal).
- Preparación del lote (PrepararLote): junta las ventanas de un lote, las normaliza (por ventana y canal,
  senal/normalizacion.py), recorta valores extremos y, solo al entrenar, aplica el aumento (red/aumento.py).
  Se hace por lote y no por ventana porque numpy normaliza 64 ventanas a la vez casi tan rápido como una.
- DataLoader (crear_cargador): pide al Dataset las ventanas de cada lote, en orden o barajadas, y se las pasa
  a PrepararLote. Lo que sale es lo que entra en la red: (lote, 22, 1.024) en float32 + etiquetas.

Por qué se normaliza aquí y no en el paquete: así es exactamente lo que hará la demo (normalizar la ventana
que acaba de llegar) y se puede cambiar la normalización sin rehacer los paquetes.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tfm_comun.red.aumento import reflejar_hemisferios
from tfm_comun.red.reproducibilidad import iniciar_trabajador
from tfm_comun.senal.normalizacion import NORMALIZACIONES


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


class PrepararLote:
    """Convierte una lista de (ventana, etiqueta) en el lote que entra en la red (collate_fn del DataLoader).

    Qué hace, en orden:
        1. Junta las ventanas: (lote, 22, 1.024) en float16.
        2. Normaliza cada ventana y canal con `normalizacion` ('robusta' o 'zscore', senal/normalizacion.py).
        3. Recorta a ±recorte (si se da): un artefacto enorme no se queda en cientos de "desviaciones" y no
           desestabiliza el entrenamiento.
        4. Si `reflejo` > 0, refleja izquierda <-> derecha esa fracción de ventanas, al azar (solo al entrenar).

    Devuelve:
        (x, y): x tensor float32 (lote, 22, 1.024); y tensor float32 (lote,) con 1 = ⏳ preictal, 0 = 💤 interictal.
    """

    def __init__(self, normalizacion='robusta', recorte=None, reflejo=0.0):
        assert normalizacion in NORMALIZACIONES, f'Normalización desconocida: {normalizacion}. Opciones: {list(NORMALIZACIONES)}'
        self.normalizacion, self.recorte, self.reflejo = normalizacion, recorte, reflejo

    def __call__(self, muestras):
        ventanas = np.stack([v for v, _ in muestras])
        ventanas = NORMALIZACIONES[self.normalizacion](ventanas).astype(np.float32, copy=False)   # la red trabaja en float32
        if self.recorte:
            np.clip(ventanas, -self.recorte, self.recorte, out=ventanas)
        x = torch.from_numpy(ventanas)
        if self.reflejo > 0:
            x = reflejar_hemisferios(x, self.reflejo)
        y = torch.as_tensor(np.array([e for _, e in muestras], dtype=np.float32))
        return x, y


def crear_cargador(dataset, preparar, tamano_lote=64, barajar=False, semilla=42, n_trabajadores=0, pin_memory=False):
    """El DataLoader que sirve los lotes de una vista.

    Parámetros:
        dataset: un VentanasEEG.  preparar: un PrepararLote (normalización, recorte, aumento).
        tamano_lote: ventanas por lote.
        barajar: True para entrenar (orden distinto en cada época, pero repetible con la semilla); False para
            validar y evaluar (mismo orden que el índice, para poder volver a juntar las predicciones con sus ventanas).
        semilla: fija el barajado y el aumento: con la misma semilla, las mismas épocas.
        n_trabajadores: procesos que preparan lotes en paralelo mientras la red entrena. 0 = en el propio notebook
            (lo más seguro en Windows y en Jupyter); con el paquete en disco y la normalización por lote suele bastar.
        pin_memory: True si se entrena en GPU NVIDIA (copia algo más rápido a la tarjeta).
    """
    generador = torch.Generator().manual_seed(semilla)
    return DataLoader(dataset, batch_size=tamano_lote, shuffle=barajar, collate_fn=preparar, generator=generador,
                      num_workers=n_trabajadores, worker_init_fn=iniciar_trabajador if n_trabajadores else None,
                      persistent_workers=n_trabajadores > 0, pin_memory=pin_memory)
