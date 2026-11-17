"""
Preparación de los datos de la red: de los paquetes de F03.01 a las secuencias listas para entrenar.

Motivo para cambiar: que cambie qué filas entran en la red o cómo se monta su entrada.

Es lo que F03.04 hace paso a paso y explicado (§3-§5), en una sola función, para que los notebooks siguientes
(F03.05 y los de las variantes) usen EXACTAMENTE los mismos datos sin repetir el código:
    1. Las filas de 🔵 train y 🟡 dev con el contexto completo más largo de F03.03 (las mismas que F03.03 y F03.04).
    2. El espectro relativo de cada ventana de 4 s que hace falta (red/entradas/), del almacén si ya está calculado.
    3. Los índices de la secuencia de cada fila (prediccion/contextos/indices.py: indices_contexto).
    4. La calibración por paciente que decidió F03.03 (prediccion/normalizacion_paciente.py).
    5. SecuenciasEspectro (red/lotes/secuencias_espectro.py), en la GPU si cabe.

F03.06 la usa con otra entrada (la log-FFT de cada segundo, como Tang et al. 2022) y un contexto fijo de 60 s:
parámetros `entrada`, `contexto_s` y `normalizacion`. Con sus valores por defecto hace lo mismo que antes.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tfm_comun.entorno.rutas import RUTA_SENALES
from tfm_comun.prediccion.almacen_caracteristicas import caracteristicas_al_dia, cargar_caracteristicas, guardar_caracteristicas
from tfm_comun.prediccion.contextos.calculo import caracteristicas_de_inicios
from tfm_comun.prediccion.contextos.disponibilidad import clave, con_contexto_completo, inicios_necesarios
from tfm_comun.prediccion.contextos.duracion import ventanas_por_contexto
from tfm_comun.prediccion.contextos.indices import indices_contexto
from tfm_comun.prediccion.normalizacion_paciente import calibracion_por_paciente
from tfm_comun.prediccion.validacion import rondas_validacion
from tfm_comun.red.entradas.bandas_segundos import bandas_por_segundo
from tfm_comun.red.entradas.espectro_relativo import espectro_relativo
from tfm_comun.red.entradas.log_fft_segundos import log_fft_por_segundo
from tfm_comun.red.entradas.rango_frecuencias import frecuencias_entrada
from tfm_comun.red.paquetes import cargar_paquete
from tfm_comun.red.lotes.secuencias_espectro import SecuenciasEspectro
from tfm_comun.senal.montaje import CANALES_TCP

FRACCION_PRUEBA = 0.1   # en modo prueba, la parte de las filas que se usa (la misma que F03.04 en modo prueba)

# Qué ve la red de cada ventana de 4 s: función que la calcula y prefijo del almacén donde se guarda
ENTRADAS = {
    'espectro_relativo': (espectro_relativo, 'espectro'),          # F03.04 y F03.05: 22 x 40 (Welch, relativo)
    'log_fft_segundo': (log_fft_por_segundo, 'logfft_segundo'),    # F03.06 (Tang): 22 x 4 s x 40 Hz
    'bandas_segundo': (bandas_por_segundo, 'bandas_segundo'),      # F03.06, diagnóstico: 22 x 4 s x 5 bandas (lo de XGBoost)
}


def preparar_datos_red(conjunto, config_01, config_03, dispositivo, modo_prueba=False, progreso=None,
                       entrada='espectro_relativo', contexto_s=None, normalizacion=None):
    """Todo lo que necesita la red para entrenar y medirse, en un solo objeto.

    Parámetros:
        conjunto: 'general', 'grupo_ausencia'...  config_01, config_03: los configuracion.json de F03.01 y F03.03.
        dispositivo: torch.device.  modo_prueba: True = un 10 % de las filas (las mismas que F03.04 en modo prueba).
        progreso: función tipo tqdm para la lectura de la señal (solo la primera vez).
        entrada: qué ve la red de cada ventana (ENTRADAS): 'espectro_relativo' (por defecto), 'log_fft_segundo' o 'bandas_segundo'.
        contexto_s: segundos de contexto de la red. None = la decisión de F03.03 (config_03['decision_red']).
        normalizacion: 'ninguna', 'por_paciente' o 'por_paciente_calibrado'. None = la decisión de F03.03.

    Devuelve:
        Un objeto con: filas (tabla), y, pacientes, rondas, validar, datos (SecuenciasEspectro), frecuencias,
        n_frecuencias (números por canal y ventana: lo que pide crear_red), n_canales, k (ventanas por secuencia),
        contexto_s, normalizacion, entrada, fs, muestras, estado_almacen.
    """
    semilla = int(config_01['semilla'])
    ventana_s, muestras = config_01['ventana_s'], int(config_01['muestras_por_ventana'])
    fs = muestras / ventana_s
    splits = [s for s in ['train', 'dev'] if s in config_01['paquetes']]           # 🟢 eval NO
    contexto_s = int(contexto_s or config_03['decision_red']['contexto_s'])
    normalizacion = normalizacion or config_03['decision_red']['normalizacion']
    assert entrada in ENTRADAS, f'Entrada desconocida: {entrada}. Opciones: {list(ENTRADAS)}'
    calcular, prefijo = ENTRADAS[entrada]
    k = ventanas_por_contexto(contexto_s, ventana_s)
    k_max = max(k, ventanas_por_contexto(max(config_03['contextos_s']), ventana_s))   # las mismas filas que F03.03

    # 1. Las filas
    todas = pd.concat([cargar_paquete(conjunto, s)[1] for s in splits], ignore_index=True)
    completo = con_contexto_completo(todas, k_max, muestras)
    if modo_prueba:
        completo &= np.random.default_rng(semilla).random(len(todas)) < FRACCION_PRUEBA
    filas = todas[completo].reset_index(drop=True)
    y = filas.etiqueta.to_numpy()
    pacientes = filas.patient.to_numpy()
    rondas = rondas_validacion(filas)
    validar = np.any([v for _, _, v in rondas], axis=0)
    for nombre, entrena, valida in rondas:
        assert not set(pacientes[entrena]) & set(pacientes[valida]), f'🔴 {nombre}: pacientes en entrenamiento y validación a la vez.'

    # 2-3. Los espectros (del almacén si ya están) y los índices de cada secuencia
    frecuencias = frecuencias_entrada(fs)
    n_canales_ = len(CANALES_TCP)
    n_frecuencias = calcular(np.ones((1, n_canales_, muestras), dtype=np.float32), fs).shape[1] // n_canales_   # números por canal y ventana
    n_canales = len(CANALES_TCP)
    necesarias = inicios_necesarios(filas, k, muestras)
    claves = clave(necesarias.ruta, necesarias.inicio)
    nombre_almacen = f'{prefijo}_{conjunto}' + ('_prueba' if modo_prueba else '')
    if caracteristicas_al_dia(nombre_almacen, 'train_dev', claves, n_canales * n_frecuencias):
        estado = 'ya estaba al día'
    else:
        estado = 'calculado'
        valores = caracteristicas_de_inicios(necesarias, RUTA_SENALES, fs, muestras, calcular=calcular, progreso=progreso)
        guardar_caracteristicas(nombre_almacen, 'train_dev', claves, valores)
    valores, claves_guardadas = cargar_caracteristicas(nombre_almacen, 'train_dev')
    indices = indices_contexto(filas, k, muestras, claves_guardadas)

    # 4. La calibración por paciente
    actual = valores[indices[:, -1]]
    if normalizacion == 'ninguna':
        grupo = np.zeros(len(filas), dtype=np.int64)
        medianas = np.zeros((1, valores.shape[1]), dtype=np.float32)
        escalas = np.ones((1, valores.shape[1]), dtype=np.float32)
    else:
        referencia = (y == 0) if normalizacion == 'por_paciente_calibrado' else np.ones(len(filas), dtype=bool)
        grupo, medianas, escalas = calibracion_por_paciente(actual, pacientes, referencia)
    del actual

    # 5. Las secuencias
    datos = SecuenciasEspectro(valores, indices, y, grupo, medianas, escalas, n_canales, dispositivo)
    return SimpleNamespace(filas=filas, y=y, pacientes=pacientes, rondas=rondas, validar=validar, datos=datos,
                           frecuencias=frecuencias, n_frecuencias=n_frecuencias, n_canales=n_canales, k=k, contexto_s=contexto_s,
                           normalizacion=normalizacion, entrada=entrada,
                           fs=fs, muestras=muestras, ventana_s=ventana_s, semilla=semilla, estado_almacen=estado,
                           ventanas_calculadas=len(claves_guardadas))
