"""Entrada "log-FFT por segundo" (F03.06-F03.08): como Tang et al. (ICLR 2022), la FFT de cada SEGUNDO de cada canal.

Motivo para cambiar: que cambie cómo se calcula la log-FFT por segundo.

Imagen: fase_03/img/clip_segundo_a_segundo.png (el clip de 60 s leído segundo a segundo)

Qué hace Tang
=============
Parte la señal en trozos de 1 s y, de cada trozo y canal, se queda con el logaritmo de la amplitud de la FFT.
La red recibe la SECUENCIA de esos segundos (60 pasos para un clip de 60 s), no un resumen por ventana.

Diferencias con Tang, y por qué
===============================
    - Frecuencias de 1 a 40 Hz (Tang: todas hasta fs/2). La señal de F02.02 viene filtrada a 0,5-40 Hz: por
      encima de 40 Hz solo queda ruido del filtro.
    - Se resta el NIVEL de cada canal en cada ventana de 4 s (la media del log en esos 4 s y 40 Hz). Tang no lo
      hace (normaliza con la media y la desviación de todo train), pero F02.05 encontró que el nivel de la señal
      es una pista tramposa (H3) y fijó como medida "normalización por ventana y canal". Así la red ve la FORMA
      del espectro y cómo cambia segundo a segundo dentro de la ventana, no lo fuerte que llega la señal.

Se guarda por ventana de 4 s (lo que maneja el almacén y prediccion/contextos/): 22 canales x 4 segundos x 40 Hz
= 3.520 números. La red (red/redes/gru_segundos.py: RedGRUSegundos) los reordena en la secuencia de segundos.
"""

import numpy as np

from tfm_comun.red.entradas.rango_frecuencias import ENERGIA_MINIMA, FRECUENCIA_MAX_HZ, FRECUENCIA_MIN_HZ, POTENCIA_MINIMA


SEGUNDO_S = 1.0               # trozo de la FFT (resolución de 1 Hz)


def segundos_por_ventana(ventana_s=4):
    """Cuántos trozos de 1 s caben en una ventana: 4 con ventanas de 4 s."""
    return int(round(ventana_s / SEGUNDO_S))


def log_fft_por_segundo(ventanas, fs, minima=FRECUENCIA_MIN_HZ, maxima=FRECUENCIA_MAX_HZ, quitar_nivel=True):
    """Log-amplitud de la FFT de cada segundo de cada canal (como Tang et al., 2022), aplanada por ventana.

    Qué hace:
        1. Parte cada ventana en trozos de 1 s (4 en una ventana de 4 s) sin solape.
        2. FFT de cada trozo y canal; amplitud |X| de las frecuencias de `minima` a `maxima` Hz (40 con 1-40 Hz).
        3. log10 con un suelo (la amplitud de 0 no tiene logaritmo).
        4. Si quitar_nivel: a cada canal se le resta su media en la ventana (4 s x 40 Hz). Ver arriba (H3 de F02.05).
        Los canales sin señal (rellenos con ceros en F02.02) quedan NaN.

    Parámetros:
        ventanas: array (ventanas, canales, muestras) en µV.  fs: muestras por segundo (256).

    Devuelve:
        Array float16 (ventanas, canales x segundos x frecuencias). Con 22 canales, 4 s y 40 Hz: 3.520 números.
        Para volver a la forma: .reshape(-1, 22, 4, 40) (canal, segundo, frecuencia).
    """
    ventanas = np.asarray(ventanas, dtype=np.float32)
    n_ventanas, n_canales, n_muestras = ventanas.shape
    paso = int(round(SEGUNDO_S * fs))
    n_segundos = n_muestras // paso
    trozos = ventanas[..., :n_segundos * paso].reshape(n_ventanas, n_canales, n_segundos, paso)
    amplitud = np.abs(np.fft.rfft(trozos, axis=-1))
    frecuencias = np.fft.rfftfreq(paso, d=1 / fs)
    elegidas = (frecuencias >= minima - 1e-9) & (frecuencias <= maxima + 1e-9)
    amplitud = amplitud[..., elegidas]                                              # (ventanas, canales, segundos, frecuencias)
    sin_senal = (ventanas.astype(np.float64) ** 2).sum(axis=-1) <= ENERGIA_MINIMA   # (ventanas, canales)
    salida = np.log10(np.maximum(amplitud, POTENCIA_MINIMA))
    if quitar_nivel:
        salida = salida - salida.mean(axis=(2, 3), keepdims=True)
    salida[sin_senal] = np.nan
    return salida.reshape(n_ventanas, -1).astype(np.float16)   # float16: la mitad de memoria (un log10 no necesita más precisión)
