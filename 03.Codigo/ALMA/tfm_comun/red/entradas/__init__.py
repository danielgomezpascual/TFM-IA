"""Entradas de la red: qué números recibe la red de cada ventana de 4 s. Una representación por fichero.

Imagen: fase_03/img/tres_entradas.png (las tres entradas, lado a lado)

    rango_frecuencias.py -> las frecuencias que entran (1-40 Hz) y los suelos numéricos comunes a todas las entradas.
    espectro_relativo.py -> Welch relativo por ventana: 22 x 40 (F03.04-F03.05).
    log_fft_segundos.py  -> log-amplitud de la FFT de cada segundo, como Tang: 22 x 4 s x 40 Hz (F03.06-F03.08).
    bandas_segundos.py   -> potencia relativa en 5 bandas de cada segundo, lo que ve XGBoost: 22 x 4 s x 5 (F03.06, diagnóstico).

Todas trabajan con arrays de numpy (sin MNE ni PyTorch): sirven igual para el TFM y para la demo.
Todas reciben (ventanas, canales, muestras) en µV y devuelven (ventanas, números) aplanado, canal a canal.
Los canales sin señal (rellenos con ceros en F02.02) quedan como NaN: la red los cambia por 0 DESPUÉS de normalizar
por paciente (red/lotes/secuencias_espectro.py), es decir, "como lo normal en este paciente".
"""



