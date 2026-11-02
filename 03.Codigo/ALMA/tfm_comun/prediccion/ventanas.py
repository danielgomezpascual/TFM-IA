"""
Ventanas: cómo se corta la señal en trozos cortos y a qué modelo pertenece cada trozo.

Motivo para cambiar: que cambie el tamaño de las ventanas, su solape o la regla que decide si una
ventana preictal entra en un modelo. Las funciones de corte trabajan con arrays de numpy (sin MNE) y
sirven igual para la demo: en directo se cortan ventanas de la señal que va llegando.

Imagen: tfm_comun/imagenes/ventanas_corte.png (cómo se cortan las ventanas antes de una crisis)

Qué es una ventana
==================
El modelo no ve un tramo entero (minutos), sino trozos cortos de VENTANA_S segundos: una matriz de
22 canales x 1.024 muestras (4 s a 256 Hz). Cada ventana hereda la clase de su tramo.

Por qué 4 s (síntesis del estado del arte, Google Doc "Síntesis estado del arte TFM")
- Las ausencias duran ~8,6 s de media: con ventanas de 12 s o más se diluyen; se recomiendan 1-5 s.
- En TUSZ se usan 1 s (FFT, Tang 2021), 2 s (Albaqami 2022 y 2023) y 5 s (Fan 2024); en predicción,
  5 s con 50 % de solape (Li/STAN 2025) o sin solape (Koutsouvelis 2024).
- 4 s cae en ese rango, cabe varias veces en los 50 s de preictal de las ausencias y son 1.024 muestras
  (potencia de 2: va bien para la FFT). Se puede cambiar aquí y comparar en la Fase 3 (Roy 2020 barre 1-16 s).

Solape (fracción de la ventana que comparte con la siguiente)
- 💤 interictal: 0 (sin solape). Sobra señal: hay unas 18 h de interictal por cada hora de preictal.
- ⏳ preictal y ⚡ ictal: 0,5 (50 %). Hay poca señal, así que se aprovecha más (como STAN 2025 y
  Georgis-Yap 2023, que solapan el preictal). No inventa datos: son trozos reales desplazados 2 s.

Alineación
- Las ventanas preictales se cortan DESDE EL FINAL del tramo (que acaba justo donde empieza el SPH), para
  que la última ventana termine exactamente en el límite del SPH y todos los modelos compartan ventanas.
- Las interictales e ictales se cortan desde el principio.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tfm_comun.senal.remuestreo import FS_OBJETIVO

VENTANA_S = 4                                                     # segundos por ventana
SOLAPE_POR_CLASE = {'interictal': 0.0, 'preictal': 0.5, 'ictal': 0.5}
DESDE_EL_FINAL = {'interictal': False, 'preictal': True, 'ictal': False}


def muestras_por_ventana(ventana_s=VENTANA_S, fs=FS_OBJETIVO):
    """Muestras de una ventana: 4 s x 256 Hz = 1.024."""
    return int(round(ventana_s * fs))


def paso_en_muestras(clase, ventana_s=VENTANA_S, fs=FS_OBJETIVO):
    """Distancia entre el inicio de una ventana y el de la siguiente: con 50 % de solape, media ventana."""
    return max(1, int(round(muestras_por_ventana(ventana_s, fs) * (1 - SOLAPE_POR_CLASE[clase]))))


def inicios_ventanas(n_muestras, tamano, paso, desde_el_final=False):
    """Muestra en la que empieza cada ventana completa que cabe en un tramo de n_muestras.

    Parámetros:
        n_muestras: longitud del tramo.
        tamano: muestras por ventana. paso: muestras entre inicios.
        desde_el_final: si es True, la última ventana acaba justo al final del tramo (las sobras quedan
            al principio); si es False, la primera empieza en 0 (las sobras quedan al final).
    Devuelve:
        Array de enteros, en orden creciente (vacío si el tramo es más corto que una ventana).
    """
    if n_muestras < tamano:
        return np.array([], dtype=int)
    inicios = np.arange(0, n_muestras - tamano + 1, paso)
    if desde_el_final:
        inicios = inicios + (n_muestras - tamano - inicios[-1])
    return inicios


def ventanas_de_senal(senal, tamano, inicios):
    """Matriz (canales x muestras) -> array (ventanas x canales x tamano) con las ventanas que empiezan en
    `inicios`. No copia datos (es una vista): sirve igual para un tramo guardado o para la demo."""
    vista = sliding_window_view(np.asarray(senal), tamano, axis=-1)   # (canales, posiciones, tamano)
    return np.moveaxis(vista[:, np.asarray(inicios)], 1, 0)


def dentro_del_preictal(inicio_s, fin_s, inicio_crisis_s, umbral_s, sph_s):
    """¿La ventana [inicio_s, fin_s] cae entera dentro del preictal de un modelo?

    El preictal de un modelo va desde (inicio de la crisis - umbral) hasta (inicio de la crisis - SPH).
    Acepta números o arrays (vectorizado). Ejemplo, modelo general (umbral 240 s, SPH 60 s) y crisis en
    el segundo 1.000: valen las ventanas entre los segundos 760 y 940.
    """
    return (np.asarray(inicio_s) >= np.asarray(inicio_crisis_s) - np.asarray(umbral_s) - 1e-6) & \
           (np.asarray(fin_s) <= np.asarray(inicio_crisis_s) - np.asarray(sph_s) + 1e-6)
