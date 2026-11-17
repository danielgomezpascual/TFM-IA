"""Entrada "bandas por segundo" (F03.06 con ENTRADA = 'bandas_segundo', diagnóstico): lo que ve XGBoost, segundo a segundo.

Motivo para cambiar: que cambie cómo se calculan las bandas de cada segundo.

Para qué sirve
==============
Con la log-FFT (880 números por segundo) la red de F03.06 se aprendía a los pacientes y no superaba a XGBoost.
Esta entrada da a la misma GRU lo que funciona en XGBoost (prediccion/caracteristicas.py: log10 de la potencia
RELATIVA por banda y canal), calculado en cada segundo: 22 canales x 5 bandas = 110 números por segundo, 8 veces
menos. Si así la red iguala o supera a XGBoost, el problema era el tamaño de la entrada, no la red.
Al ser relativa, el nivel de la señal ya no entra (H3 de F02.05).
"""

import numpy as np

from tfm_comun.red.entradas.log_fft_segundos import SEGUNDO_S


def bandas_por_segundo(ventanas, fs):
    """Potencia relativa por banda (log10) de cada segundo de cada canal, aplanada por ventana.

    Qué hace: parte cada ventana en trozos de 1 s y a cada trozo le aplica calcular_caracteristicas()
    (prediccion/caracteristicas.py, las mismas 5 bandas que XGBoost). Los canales sin señal quedan NaN.

    Parámetros:
        ventanas: array (ventanas, canales, muestras) en µV.  fs: muestras por segundo (256).

    Devuelve:
        Array float32 (ventanas, canales x segundos x bandas): con 22 canales, 4 s y 5 bandas, 440 números.
        Mismo orden que log_fft_por_segundo (canal, segundo, banda): .reshape(-1, 22, 4, 5).
    """
    from tfm_comun.prediccion.caracteristicas import calcular_caracteristicas   # aquí para no crear un ciclo de imports

    ventanas = np.asarray(ventanas, dtype=np.float32)
    n_ventanas, n_canales, n_muestras = ventanas.shape
    paso = int(round(SEGUNDO_S * fs))
    n_segundos = n_muestras // paso
    trozos = ventanas[..., :n_segundos * paso].reshape(n_ventanas, n_canales, n_segundos, paso)
    trozos = trozos.transpose(0, 2, 1, 3).reshape(n_ventanas * n_segundos, n_canales, paso)   # cada segundo como una "ventana"
    valores = calcular_caracteristicas(trozos, fs)                                              # (ventanas x segundos, canales x bandas)
    valores = valores.reshape(n_ventanas, n_segundos, n_canales, -1).transpose(0, 2, 1, 3)      # (ventanas, canales, segundos, bandas)
    return np.ascontiguousarray(valores).reshape(n_ventanas, -1).astype(np.float32)
