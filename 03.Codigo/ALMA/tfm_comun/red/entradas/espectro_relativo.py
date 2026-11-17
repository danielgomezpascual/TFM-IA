"""Entrada "espectro relativo" (F03.04-F03.05): el espectro de Welch de cada canal en una ventana de 4 s, relativo y en log.

Motivo para cambiar: que cambie cómo se calcula el espectro relativo.

Imagen: fase_03/img/de_ventana_a_espectro.png (de la ventana al espectro relativo que entra en la red)

Qué recibe la red: el ESPECTRO RELATIVO de cada canal (22 canales x 40 frecuencias)
=================================================================================
1. Espectro de cada canal con el método de Welch, igual que en F03.02 (prediccion/caracteristicas.py): trozos
   de 1 s solapados a la mitad -> energía en cada frecuencia, de 1 en 1 Hz.
2. Se quedan las frecuencias de 1 a 40 Hz (40 números por canal): red/entradas/rango_frecuencias.py.
3. RELATIVO: cada frecuencia se divide por la energía total del canal (suman 1). Como en F03.02, así el
   nivel de la señal (que depende del equipo y del paciente, pista tramposa H3 de F02.05) no entra.
4. log10, con un suelo, para que 0,0001 y 0,3 queden en una escala parecida.

Por qué el espectro y no la señal en bruto (22 x 1.024):
    - Es lo que funcionó en F03.02 y F03.03 (potencia relativa), pero con 40 frecuencias en vez de 5 bandas:
      la red decide sola qué frecuencias importan, en vez de usar las bandas clásicas fijas.
    - La normalización por paciente de F03.03 (restar lo normal en cada paciente) necesita números que se puedan
      comparar entre ventanas; con la señal en bruto no tiene sentido (la onda nunca se repite igual).
    - Ocupa 25 veces menos (880 números frente a 22.528): todo el contexto de 60 s cabe en la memoria de la GPU.
"""

import numpy as np
from scipy.signal import welch

from tfm_comun.red.entradas.rango_frecuencias import (ENERGIA_MINIMA, FRECUENCIA_MAX_HZ, FRECUENCIA_MIN_HZ, POTENCIA_MINIMA,
                                                      SEGMENTO_S)


def espectro_relativo(ventanas, fs, minima=FRECUENCIA_MIN_HZ, maxima=FRECUENCIA_MAX_HZ):
    """El espectro relativo (log10) de cada canal de cada ventana, aplanado: lo que recibe la red.

    Qué hace:
        1. Welch con trozos de 1 s solapados a la mitad (resolución de 1 Hz).
        2. Se queda con las frecuencias de `minima` a `maxima` Hz.
        3. Divide por la energía total del canal en ese rango (potencia relativa, suma 1).
        4. log10 con un suelo de POTENCIA_MINIMA. Los canales sin señal quedan NaN.

    Parámetros:
        ventanas: array (ventanas, canales, muestras), en µV (no hace falta normalizar).  fs: muestras por segundo.

    Devuelve:
        Array float32 (ventanas, canales x frecuencias), canal a canal: las 40 primeras columnas son FP1-F7 de 1 a 40 Hz.
        Con 22 canales y 40 frecuencias: 880 números por ventana. Para volver a (ventanas, 22, 40): .reshape(-1, 22, 40).
    """
    ventanas = np.asarray(ventanas, dtype=np.float32)
    nperseg = int(round(SEGMENTO_S * fs))
    frecuencias, espectro = welch(ventanas, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    elegidas = (frecuencias >= minima - 1e-9) & (frecuencias <= maxima + 1e-9)
    espectro = espectro[..., elegidas]
    total = espectro.sum(axis=-1, keepdims=True)
    relativo = np.where(total > ENERGIA_MINIMA, espectro / np.maximum(total, ENERGIA_MINIMA), np.nan)
    salida = np.log10(np.maximum(relativo, POTENCIA_MINIMA))                    # NaN se queda NaN
    return salida.reshape(len(salida), -1).astype(np.float32)
