"""
Preparación de los datos de la red: de los paquetes de F03.01 a las secuencias listas para entrenar.

Motivo para cambiar: que cambie qué filas entran en la red o cómo se monta su entrada.

Es lo que F03.04 hace paso a paso y explicado (§3-§5), en una sola función, para que los notebooks siguientes
(F03.05 y los de las variantes) usen EXACTAMENTE los mismos datos sin repetir el código:
    1. Las filas de 🔵 train y 🟡 dev con el contexto completo más largo de F03.03 (las mismas que F03.03 y F03.04).
    2. El espectro relativo de cada ventana de 4 s que hace falta (red/entradas/), del almacén si ya está calculado.
    3. Los índices de la secuencia de cada fila (prediccion/contextos/indices.py: indices_contexto).
    4. La calibración por paciente que decidió F03.03, con sus mismas reglas y nombres
       (prediccion/normalizacion_paciente.py: reserva_de_filas y calibracion_para_red).
    4b. Solo las filas que usa F03.03 (o F03.03b): sin las 💤 reservadas para calibrar ni las de pacientes (o sesiones)
        sin calibración. Así la red se entrena y se mide con las mismas ventanas que XGBoost.
    4c. El MUESTREO y el MONTAJE de entrenamiento de F03.03b (prediccion/atajo_grabacion.py): qué ventanas se usan al entrenar.
La decisión (contexto, normalización y muestreo) es la de F03.03b si se ejecutó; si no, la de F03.03
(prediccion/atajo_grabacion.py: leer_decision_red).
    5. SecuenciasEspectro (red/lotes/secuencias_espectro.py), en la GPU si cabe.

F03.06 la usa con otra entrada (la log-FFT de cada segundo, como Tang et al. 2022) y un contexto fijo de 60 s:
parámetros `entrada`, `contexto_s` y `normalizacion`. Con sus valores por defecto hace lo mismo que antes.
"""

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from tfm_comun.entorno.rutas import RUTA_SENALES
from tfm_comun.prediccion.almacen_caracteristicas import caracteristicas_al_dia, cargar_caracteristicas, guardar_caracteristicas
from tfm_comun.prediccion.contextos.calculo import caracteristicas_de_inicios
from tfm_comun.prediccion.contextos.disponibilidad import clave, con_contexto_completo, inicios_necesarios
from tfm_comun.prediccion.contextos.duracion import ventanas_por_contexto
from tfm_comun.prediccion.contextos.indices import indices_contexto
from tfm_comun.prediccion.atajo_grabacion import leer_decision_red, mascara_montaje, mascara_muestreo, posicion_respecto_crisis
from tfm_comun.prediccion.normalizacion_paciente import calibracion_para_red, ventanas_del_estudio
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
                       entrada='espectro_relativo', contexto_s=None, normalizacion=None, muestreo=None, montaje_entrenamiento=None):
    """Todo lo que necesita la red para entrenar y medirse, en un solo objeto.

    Parámetros:
        conjunto: 'general', 'grupo_ausencia'...  config_01, config_03: los configuracion.json de F03.01 y F03.03.
        dispositivo: torch.device.  modo_prueba: True = un 10 % de las filas (las mismas que F03.04 en modo prueba).
        progreso: función tipo tqdm para la lectura de la señal (solo la primera vez).
        entrada: qué ve la red de cada ventana (ENTRADAS): 'espectro_relativo' (por defecto), 'log_fft_segundo' o 'bandas_segundo'.
        contexto_s: segundos de contexto de la red. None = la decisión de F03.03 (config_03['decision_red']).
        normalizacion: 'ninguna', 'por_paciente', 'por_paciente_reserva' o 'por_sesion_reserva'. None = la decisión
            (F03.03b si existe; si no, F03.03). La calibración antigua con fuga ('por_paciente_calibrado') ya no se acepta.
        muestreo: qué 💤 se usan al entrenar ('todas', 'misma_sesion', 'misma_sesion_antes'). None = la decisión.
        montaje_entrenamiento: con qué montajes originales se entrena ('todos' o '01_tcp_ar'). None = la decisión.

    Devuelve:
        Un objeto con: filas (tabla), y, pacientes, rondas, validar, datos (SecuenciasEspectro), frecuencias,
        n_frecuencias (números por canal y ventana: lo que pide crear_red), n_canales, k (ventanas por secuencia),
        contexto_s, normalizacion, muestreo, montaje_entrenamiento, origen (de dónde sale la decisión), antes (True en las ventanas de antes de
        toda crisis de su sesión: para el control estricto), entrada, fs, muestras, estado_almacen.
    """
    semilla = int(config_01['semilla'])
    ventana_s, muestras = config_01['ventana_s'], int(config_01['muestras_por_ventana'])
    fs = muestras / ventana_s
    splits = [s for s in ['train', 'dev'] if s in config_01['paquetes']]           # 🟢 eval NO
    decision = leer_decision_red(conjunto, config_03)
    contexto_s = int(contexto_s or decision['contexto_s'])
    normalizacion = normalizacion or decision['normalizacion']
    muestreo = muestreo or decision['muestreo']
    montaje_entrenamiento = montaje_entrenamiento or decision['montaje_entrenamiento']
    config_ref = decision['config_03b'] or config_03              # el notebook que fijó las ventanas
    assert entrada in ENTRADAS, f'Entrada desconocida: {entrada}. Opciones: {list(ENTRADAS)}'
    calcular, prefijo = ENTRADAS[entrada]
    k = ventanas_por_contexto(contexto_s, ventana_s)
    k_max = max(k, ventanas_por_contexto(max(config_03['contextos_s']), ventana_s))   # las mismas filas que F03.03

    # 1. Las filas con contexto completo (todas: la calibración por paciente de F03.03 se calcula con todas ellas)
    todas = pd.concat([cargar_paquete(conjunto, s)[1] for s in splits], ignore_index=True)
    completo = con_contexto_completo(todas, k_max, muestras)
    if modo_prueba:
        completo &= np.random.default_rng(semilla).random(len(todas)) < FRACCION_PRUEBA
    filas = todas[completo].reset_index(drop=True)
    reserva = config_03.get('reserva_calibracion')
    assert reserva, 'El configuracion.json de F03.03 no tiene reserva_calibracion: vuelve a ejecutar F03.03.'
    reservada, usar, grupos_calibracion = ventanas_del_estudio(filas, fs, ventana_s, reserva, normalizacion, decision['reserva_por_sesion'])

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

    # 4. La calibración por paciente (con todas las filas, como F03.03)
    grupo, medianas, escalas = calibracion_para_red(valores[indices[:, -1]], grupos_calibracion, normalizacion, reservada)

    # 4b. Desde aquí, solo las filas que usa F03.03 (sin reservadas ni pacientes sin calibración)
    filas, indices, grupo = filas[usar].reset_index(drop=True), indices[usar], grupo[usar]
    assert (grupo >= 0).all(), 'Hay filas sin calibración: revisa reserva_de_filas.'
    y = filas.etiqueta.to_numpy()
    pacientes = filas.patient.to_numpy()
    rondas = rondas_validacion(filas)
    validar = np.any([v for _, _, v in rondas], axis=0)
    for nombre, entrena, valida in rondas:
        assert not set(pacientes[entrena]) & set(pacientes[valida]), f'🔴 {nombre}: pacientes en entrenamiento y validación a la vez.'

    # 4c. El muestreo de entrenamiento (solo cambia con qué se ENTRENA; la validación es siempre la misma)
    antes = posicion_respecto_crisis(filas, fs).antes_de_toda_crisis.to_numpy()
    ficheros = filas.file.to_numpy()
    rondas = [(nombre, mascara_montaje(mascara_muestreo(entrena, y, pacientes, ficheros, antes, muestreo), ficheros, montaje_entrenamiento), valida)
              for nombre, entrena, valida in rondas]
    mismas_filas_que_f0303 = k_max == ventanas_por_contexto(max(config_03['contextos_s']), ventana_s)   # con más contexto, hay menos filas
    if not modo_prueba and mismas_filas_que_f0303 and len(filas) != config_ref.get('ventanas_estudiadas', len(filas)):
        warnings.warn(f"🔴 {decision['origen']} estudió {config_ref['ventanas_estudiadas']} ventanas y aquí hay {len(filas)}: "
                      '¿se volvió a ejecutar F03.01 después?')

    # 5. Las secuencias
    datos = SecuenciasEspectro(valores, indices, y, grupo, medianas, escalas, n_canales, dispositivo)
    return SimpleNamespace(filas=filas, y=y, pacientes=pacientes, rondas=rondas, validar=validar, datos=datos,
                           frecuencias=frecuencias, n_frecuencias=n_frecuencias, n_canales=n_canales, k=k, contexto_s=contexto_s,
                           normalizacion=normalizacion, muestreo=muestreo, montaje_entrenamiento=montaje_entrenamiento, origen=decision['origen'], antes=antes, entrada=entrada,
                           fs=fs, muestras=muestras, ventana_s=ventana_s, semilla=semilla, estado_almacen=estado,
                           ventanas_calculadas=len(claves_guardadas))
