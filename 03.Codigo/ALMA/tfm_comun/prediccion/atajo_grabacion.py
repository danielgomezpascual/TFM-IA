"""
Atajo de la grabación: dónde está cada 💤 respecto a las crisis de su sesión, y cómo quitarle al modelo la pista
de "esta grabación es de las que tienen crisis".

Motivo para cambiar: que cambie cómo se sitúa cada 💤 respecto a las crisis, qué 💤 se usan para entrenar
(muestreo) o qué controles de sesión se miden.

Lo usan F03.03b (donde nace), F03.04 y red/preparacion.py (F03.05-F03.08).

El problema (visto en F03.03 y F03.04, 2026-11-19)
==================================================
Las ⏳ salen siempre de sesiones con crisis; casi todas las 💤, de OTRAS sesiones. Un modelo puede acertar
reconociendo la grabación (montaje, equipo, tipo de registro) sin saber nada de crisis. En F03.04 la red, con
💤 de la misma sesión, cayó por DEBAJO del azar. Dos sospechas:
    1. reconoce la sesión (la pista de la grabación);
    2. dentro de la sesión, las 💤 que quedan son a menudo de DESPUÉS de una crisis (más allá de los 5 min de
       margen postictal de umbrales.py el EEG puede seguir alterado), y "alterado" lo ha aprendido como ⏳.

Qué ofrece este módulo
======================
    crisis_de_sesiones()   las crisis de TUSZ (inventario de F01.01: crisis.csv) con su sesión y su orden.
    posicion_respecto_crisis()  para cada ventana: su sesión, si la sesión tiene crisis, si es de ANTES de toda
                           crisis de su sesión y los segundos desde el final de la última crisis (si la hubo).
    controles_estrictos()  los controles de sesión de prediccion/control_sesion.py + dos nuevos:
                           'misma sesión, antes de la crisis' (como control_misma_sesion_estricta de F02.05,
                           pero en 🟡 dev) y 'misma sesión, después de una crisis' (para ver el efecto postictal).
    mascara_montaje()      qué filas de entrenamiento se usan según su montaje original ('todos' o solo '01_tcp_ar'):
                           en 🔵 train el montaje va con las crisis (03_tcp_ar_a sobre todo en sesiones con crisis) y en
                           🟡 dev no, así que un modelo puede aprender "montaje" en vez de "crisis" (F03.03b, 2026-11-19).
    mascara_muestreo()     qué filas de entrenamiento se usan con cada muestreo:
                           'todas' · 'misma_sesion' (las 💤 solo de sesiones donde su paciente tiene ⏳) ·
                           'misma_sesion_antes' (además, solo 💤 de antes de toda crisis de la sesión).
    leer_decision_red()    la decisión que recibe la red: la de F03.03b si existe; si no, la de F03.03.

Supuesto (el mismo de F02.05, auditoria/sesiones.py): los ficheros de una sesión (t000, t001...) van en orden
de tiempo. No se sabe cuánto tiempo hay entre ellos.
"""

import json

import numpy as np
import pandas as pd

from tfm_comun.entorno.rutas import RUTA_OUTPUTS_FASE_01, RUTA_OUTPUTS_FASE_03
from tfm_comun.prediccion.contextos.disponibilidad import inicio_tramo_s

RUTA_CRISIS = RUTA_OUTPUTS_FASE_01 / '01_inventario' / 'crisis.csv'
RUTA_EDFS = RUTA_OUTPUTS_FASE_01 / '01_inventario' / 'edfs.csv'
MUESTREOS = {
    'todas': 'todas las 💤 de entrenamiento (como hasta ahora)',
    'misma_sesion': 'solo 💤 de sesiones en las que su paciente tiene ⏳',
    'misma_sesion_antes': 'solo 💤 de sesiones con ⏳ y de ANTES de toda crisis de la sesión',
}
NOMBRE_CARPETA_03B = '03b_atajo_grabacion'
MONTAJES_ENTRENAMIENTO = {
    'todos': 'todos los montajes originales',
    '01_tcp_ar': 'solo ficheros con montaje original 01_tcp_ar (el más frecuente): sin la pista del montaje',
}


def sesion_y_token(ficheros):
    """'aaaaaajy_s003_t004' -> ('aaaaaajy_s003', 4). Devuelve dos arrays."""
    f = pd.Series(np.asarray(ficheros)).astype(str)
    partes = f.str.rsplit('_', n=1)
    return partes.str[0].to_numpy(), partes.str[1].str[1:].astype(int).to_numpy()


def crisis_de_sesiones(ruta=RUTA_CRISIS):
    """Las crisis del inventario de F01.01, con su sesión ('aaaaaajy_s003'), token (int), inicio y fin (s del fichero)."""
    assert ruta.exists(), f'No está {ruta}. Ejecuta antes F01.01 (fase_01/01_inventario_tusz.ipynb).'
    crisis = pd.read_csv(ruta)
    sesion, token = sesion_y_token(crisis.file)
    return pd.DataFrame({'sesion': sesion, 'token': token, 'file': crisis.file.astype(str),
                         'inicio_s': crisis.start.astype(float), 'fin_s': crisis.stop.astype(float)})


def montaje_de_ficheros(ficheros, ruta=RUTA_EDFS):
    """El montaje original de cada fichero ('01_tcp_ar', '02_tcp_le'...), del inventario de F01.01 (NaN si no está)."""
    edfs = pd.read_csv(ruta, usecols=['file', 'montage']).drop_duplicates('file').set_index('file').montage
    return pd.Series(np.asarray(ficheros)).astype(str).map(edfs).to_numpy()


def posicion_respecto_crisis(filas, fs, crisis=None):
    """Dónde está cada ventana respecto a las crisis de su sesión.

    Parámetros:
        filas: tabla de ventanas (file, ruta, muestra_inicio).  fs: muestras por segundo.
        crisis: tabla de crisis_de_sesiones() (None = se lee del inventario).

    Devuelve:
        DataFrame (mismo orden que filas) con:
            sesion                     'aaaaaajy_s003'
            sesion_con_crisis          True si algún fichero de la sesión tiene alguna crisis
            antes_de_toda_crisis       True si en su sesión no hubo NINGUNA crisis antes de esta ventana
                                       (ni en ficheros anteriores ni antes en su mismo fichero)
            segundos_desde_crisis      segundos desde el FINAL de la última crisis de su mismo fichero (NaN si no la hay)
    """
    crisis = crisis_de_sesiones() if crisis is None else crisis
    sesion, token = sesion_y_token(filas.file)
    segundo = inicio_tramo_s(filas.ruta) + filas.muestra_inicio.to_numpy() / fs
    primer_token_con_crisis = crisis.groupby('sesion').token.min()
    sesion_con_crisis = pd.Series(sesion).isin(primer_token_con_crisis.index).to_numpy()
    primer_token = pd.Series(sesion).map(primer_token_con_crisis).to_numpy(dtype=float)     # NaN si la sesión no tiene crisis

    # Crisis anteriores en el MISMO fichero: la primera que empieza en ese fichero y el final de la última antes de la ventana
    antes_en_fichero = np.zeros(len(filas), dtype=bool)
    desde = np.full(len(filas), np.nan)
    por_fichero = {f: g.sort_values('inicio_s') for f, g in crisis.groupby('file')}
    ficheros = filas.file.astype(str).to_numpy()
    for f in np.unique(ficheros):
        if f not in por_fichero:
            continue
        g = por_fichero[f]
        idx = np.where(ficheros == f)[0]
        t = segundo[idx]
        empezadas = np.searchsorted(g.inicio_s.to_numpy(), t, side='right')       # crisis que empiezan antes de la ventana
        antes_en_fichero[idx] = empezadas > 0
        fines = np.maximum.accumulate(g.fin_s.to_numpy())
        tiene = empezadas > 0
        desde[idx[tiene]] = t[tiene] - fines[empezadas[tiene] - 1]

    hubo_antes = (token > np.nan_to_num(primer_token, nan=np.inf)) | antes_en_fichero
    return pd.DataFrame({'sesion': sesion, 'sesion_con_crisis': sesion_con_crisis,
                         'antes_de_toda_crisis': ~hubo_antes, 'segundos_desde_crisis': desde})


def _sesiones_con_preictal(mascara, etiqueta, pacientes, sesiones):
    """(paciente, sesión) de las ⏳ dentro de `mascara`."""
    m = np.asarray(mascara, dtype=bool) & (np.asarray(etiqueta) == 1)
    return set(zip(np.asarray(pacientes)[m], np.asarray(sesiones)[m]))


def _en(pares, pacientes, sesiones):
    return np.fromiter(((p, s) in pares for p, s in zip(pacientes, sesiones)), dtype=bool, count=len(pacientes))


def controles_estrictos(validar, etiqueta, pacientes, ficheros, antes_de_toda_crisis):
    """Los controles de sesión, con dos más que separan las 💤 de antes y de después de una crisis.

    Las ⏳ son siempre las mismas; cambian las 💤:
        'todas las 💤'                       todas las de validación
        'misma sesión'                       solo las de las sesiones donde su paciente tiene ⏳
        'misma sesión, antes de la crisis'   de esas, solo las de ANTES de toda crisis de la sesión (sin postictal posible)
        'misma sesión, después de una crisis' de esas, solo las de después de alguna crisis (diagnóstico del postictal)
        'mismo fichero'                      solo las de los ficheros donde su paciente tiene ⏳
    """
    validar, etiqueta = np.asarray(validar, dtype=bool), np.asarray(etiqueta)
    pacientes = np.asarray(pacientes)
    antes = np.asarray(antes_de_toda_crisis, dtype=bool)
    sesiones, _ = sesion_y_token(ficheros)
    ficheros = np.asarray(ficheros).astype(str)
    misma_sesion = _en(_sesiones_con_preictal(validar, etiqueta, pacientes, sesiones), pacientes, sesiones)
    mismo_fichero = _en(_sesiones_con_preictal(validar, etiqueta, pacientes, ficheros), pacientes, ficheros)
    preictal = validar & (etiqueta == 1)
    interictal = validar & (etiqueta == 0)
    return {'todas las 💤': validar,
            'misma sesión': preictal | (interictal & misma_sesion),
            'misma sesión, antes de la crisis': preictal | (interictal & misma_sesion & antes),
            'misma sesión, después de una crisis': preictal | (interictal & misma_sesion & ~antes),
            'mismo fichero': preictal | (interictal & mismo_fichero)}


CONTROL_ESTRICTO = 'misma sesión, antes de la crisis'


def mascara_muestreo(entrena, etiqueta, pacientes, ficheros, antes_de_toda_crisis, muestreo):
    """Qué filas de entrenamiento se usan con un muestreo (MUESTREOS). Las ⏳ se quedan siempre.

    Parámetros:
        entrena: True en las filas de entrenamiento de la ronda.  etiqueta, pacientes, ficheros: de cada fila.
        antes_de_toda_crisis: de posicion_respecto_crisis().  muestreo: 'todas', 'misma_sesion' o 'misma_sesion_antes'.

    Devuelve:
        Array True/False del largo de las filas (subconjunto de `entrena`).
    """
    assert muestreo in MUESTREOS, f'Muestreo desconocido: {muestreo}. Opciones: {list(MUESTREOS)}'
    entrena, etiqueta = np.asarray(entrena, dtype=bool), np.asarray(etiqueta)
    if muestreo == 'todas':
        return entrena.copy()
    pacientes = np.asarray(pacientes)
    sesiones, _ = sesion_y_token(ficheros)
    misma = _en(_sesiones_con_preictal(entrena, etiqueta, pacientes, sesiones), pacientes, sesiones)
    vale = misma if muestreo == 'misma_sesion' else misma & np.asarray(antes_de_toda_crisis, dtype=bool)
    return entrena & ((etiqueta == 1) | vale)


def mascara_montaje(entrena, ficheros, montaje_entrenamiento):
    """Qué filas de entrenamiento se usan según su montaje original (MONTAJES_ENTRENAMIENTO). Solo cambia el ENTRENAMIENTO."""
    assert montaje_entrenamiento in MONTAJES_ENTRENAMIENTO, (f'Montaje desconocido: {montaje_entrenamiento}. '
                                                           f'Opciones: {list(MONTAJES_ENTRENAMIENTO)}')
    entrena = np.asarray(entrena, dtype=bool)
    if montaje_entrenamiento == 'todos':
        return entrena.copy()
    return entrena & (montaje_de_ficheros(ficheros) == montaje_entrenamiento)


def leer_decision_red(conjunto, config_03, sufijo=''):
    """Lo que recibe la red: la decisión de F03.03b si se ejecutó; si no, la de F03.03 (con muestreo 'todas').

    Devuelve:
        dict con contexto_s, normalizacion, muestreo, montaje_entrenamiento, reserva_por_sesion (bool: si las ventanas
        excluyen también la reserva por sesión), origen ('F03.03b' o 'F03.03'), texto y config_03b (o None).
    """
    ruta = RUTA_OUTPUTS_FASE_03 / conjunto / (NOMBRE_CARPETA_03B + sufijo) / 'configuracion.json'
    if ruta.exists():
        config_03b = json.loads(ruta.read_text(encoding='utf-8'))
        assert config_03b.get('conjunto') == conjunto, f'{ruta} es de otro conjunto ({config_03b.get("conjunto")}).'
        d = config_03b['decision_red']
        return {'contexto_s': int(d['contexto_s']), 'normalizacion': d['normalizacion'], 'muestreo': d['muestreo'],
                'montaje_entrenamiento': d.get('montaje_entrenamiento', 'todos'),
                'reserva_por_sesion': bool(config_03b.get('reserva_calibracion', {}).get('por_sesion', False)),
                'origen': 'F03.03b', 'texto': d['texto'], 'config_03b': config_03b}
    d = config_03['decision_red']
    return {'contexto_s': int(d['contexto_s']), 'normalizacion': d['normalizacion'], 'muestreo': 'todas',
            'montaje_entrenamiento': 'todos', 'reserva_por_sesion': False, 'origen': 'F03.03', 'texto': d['texto'], 'config_03b': None}
