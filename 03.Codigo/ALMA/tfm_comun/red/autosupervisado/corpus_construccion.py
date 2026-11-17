"""Construcción del corpus (F03.07): EEG SIN etiquetas de los pacientes de 🔵 train, convertido en la entrada de la red
(log-FFT de cada segundo, red/entradas/log_fft_segundos.py).

Motivo para cambiar: que cambie de dónde sale la señal sin etiquetar o cómo se guarda.

Por qué hace falta
==================
La red de F03.06 se aprende de memoria a los ~90 pacientes con crisis que tiene para entrenar. En 🔵 train hay además
~400 pacientes SIN crisis (unas 400 h) que la predicción no puede usar, porque no tienen ⏳ preictal. El
preentrenamiento sí puede: no necesita etiquetas, solo señal (Tang et al., 2022: "predecir los 12 s siguientes").
F02.02 solo preprocesó los tramos que usa la predicción, así que aquí se preprocesan los EDF de 🔵 train enteros.

Qué se guarda (en RUTA_CORPUS, una sola vez para todo el TFM)
=============================================================
    corpus/<fichero>.npy     (ventanas de 4 s seguidas, 22 canales x 4 s x 40 Hz) en float16: la log-FFT de cada segundo
    corpus.csv               una fila por fichero: paciente, segundos, ventanas, estado ('hecho' o el error)

La señal pasa por la MISMA cadena que F02.02 (senal/preprocesado.py: montaje TCP, 256 Hz, filtro 0,5-40 Hz) y la
MISMA entrada que F03.06 (log_fft_por_segundo), así que la red ve en el preentrenamiento lo mismo que después.

Reglas:
    - Solo 🔵 train. 🟡 dev y 🟢 eval NUNCA (si no, el preentrenamiento habría visto a los pacientes con los que se
      mide la red: fuga).
    - Como mucho HORAS_MAX_PACIENTE por paciente (en el orden de sus ficheros): unos pocos pacientes tienen más de
      30 h y dominarían el corpus.
    - Se quitan los primeros DESCARTE_INICIO_S segundos de cada fichero (arranque del filtro, como en F02.03).
    - Se puede parar y seguir: los ficheros ya hechos (en corpus.csv) no se repiten.
"""

import time

import numpy as np
import pandas as pd

from tfm_comun.dataset.edf import leer_edf_tusz
from tfm_comun.entorno.rutas import RUTA_OUTPUTS_FASE_03
from tfm_comun.red.entradas.log_fft_segundos import log_fft_por_segundo
from tfm_comun.senal.preprocesado import preprocesar_raw
from tfm_comun.senal.remuestreo import FS_OBJETIVO


RUTA_CORPUS = RUTA_OUTPUTS_FASE_03 / 'comun' / 'preentreno' / 'corpus_logfft'


HORAS_MAX_PACIENTE = 2.0        # tope de señal por paciente


DESCARTE_INICIO_S = 10          # arranque del filtro (como F02.03)


VENTANA_S = 4                   # ventanas de 4 s seguidas, sin solape (la unidad del almacén y de la red)


TROZO_VENTANAS = 512            # ventanas por llamada a log_fft_por_segundo (para no llenar la memoria)


def plan_corpus(edfs, horas_max_paciente=HORAS_MAX_PACIENTE):
    """Qué trozo de cada EDF de 🔵 train entra en el corpus.

    Parámetros:
        edfs: la tabla edfs.csv de F01.01 (columnas split, patient, file, edf_dur_s).

    Devuelve:
        Tabla con una fila por fichero que entra: patient, file, segundos (los que se leen, desde el inicio del fichero).
        Los ficheros de cada paciente entran en orden hasta llegar a su tope; el último puede entrar recortado.
    """
    train = edfs[edfs.split == 'train'].sort_values(['patient', 'file'])
    filas, tope = [], horas_max_paciente * 3600
    for paciente, ficheros in train.groupby('patient', sort=True):
        quedan = tope
        for _, f in ficheros.iterrows():
            util = f.edf_dur_s - DESCARTE_INICIO_S
            if quedan <= 0 or util < VENTANA_S:
                continue
            leer = min(util, quedan)
            filas.append({'patient': paciente, 'file': f.file, 'segundos': DESCARTE_INICIO_S + leer})
            quedan -= leer
    return pd.DataFrame(filas)


def ventanas_log_fft(senal, fs=FS_OBJETIVO):
    """De una señal preprocesada (22 x muestras) a la log-FFT de sus ventanas de 4 s seguidas (sin las del arranque).

    Devuelve:
        Array float16 (ventanas, 22 x 4 x 40).
    """
    tamano = int(VENTANA_S * fs)
    inicio = int(DESCARTE_INICIO_S * fs)
    n = (senal.shape[1] - inicio) // tamano
    if n <= 0:
        return np.empty((0, 0), dtype=np.float16)
    ventanas = senal[:, inicio:inicio + n * tamano].reshape(senal.shape[0], n, tamano).transpose(1, 0, 2)
    partes = [log_fft_por_segundo(ventanas[a:a + TROZO_VENTANAS], fs) for a in range(0, n, TROZO_VENTANAS)]
    return np.concatenate(partes)


def construir_corpus(plan, edf_de, progreso=None, ruta=RUTA_CORPUS):
    """Lee, preprocesa y convierte cada EDF del plan. Se puede interrumpir: al volver, sigue donde lo dejó.

    Parámetros:
        plan: de plan_corpus().  edf_de: {fichero: ruta del EDF} (entorno/rutas.py: indice_edf()).
        progreso: función tipo tqdm (opcional).

    Devuelve:
        La tabla corpus.csv al día (una fila por fichero del plan ya intentado).
    """
    (ruta / 'corpus').mkdir(parents=True, exist_ok=True)
    ruta_csv = ruta / 'corpus.csv'
    hechos = pd.read_csv(ruta_csv) if ruta_csv.exists() else pd.DataFrame(columns=['patient', 'file', 'segundos', 'ventanas', 'estado', 'minutos'])
    pendientes = plan[~plan.file.isin(hechos.file)]
    recorrido = progreso(pendientes.itertuples(), total=len(pendientes)) if progreso else pendientes.itertuples()
    nuevas = []
    for f in recorrido:
        inicio = time.time()
        try:
            raw = leer_edf_tusz(edf_de[f.file], fin_s=float(f.segundos))
            senal, _ = preprocesar_raw(raw)
            valores = ventanas_log_fft(senal)
            tmp = ruta / 'corpus' / f'{f.file}_tmp.npy'
            np.save(tmp, valores)
            tmp.replace(ruta / 'corpus' / f'{f.file}.npy')
            estado, n = 'hecho', len(valores)
        except Exception as error:                     # un EDF roto no para el corpus: se apunta y se sigue
            estado, n = f'error: {type(error).__name__}: {error}'[:200], 0
        nuevas.append({'patient': f.patient, 'file': f.file, 'segundos': f.segundos, 'ventanas': n, 'estado': estado,
                       'minutos': (time.time() - inicio) / 60})
        if len(nuevas) % 25 == 0:                       # se apunta cada 25 ficheros: si se corta, se pierde poco
            hechos = pd.concat([hechos, pd.DataFrame(nuevas)], ignore_index=True)
            hechos.to_csv(ruta_csv, index=False)
            nuevas = []
    if nuevas:
        hechos = pd.concat([hechos, pd.DataFrame(nuevas)], ignore_index=True)
        hechos.to_csv(ruta_csv, index=False)
    return hechos
