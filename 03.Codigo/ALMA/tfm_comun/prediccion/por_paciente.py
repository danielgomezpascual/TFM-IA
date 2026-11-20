"""
Predicción por paciente: un modelo para cada paciente, entrenado con unas sesiones suyas y medido en otra.

Motivo para cambiar: que cambie cómo se eligen los pacientes, qué 💤 se usan, cómo se valida o cómo se decide
si un paciente "se puede predecir".

Lo usa fase_03/03c_por_paciente.ipynb (F03.03c).

Por qué (F03.03b, 2026-11-19)
=============================
Con un solo modelo para todos los pacientes no queda señal preictal en TUSZ cuando se quita la pista de la
grabación. Pero lo que mejor funciona en la literatura de predicción es un modelo POR PACIENTE: el "aviso" de
cada persona es distinto. Aquí se prueba eso con los pacientes de TUSZ que tienen varias crisis en varias sesiones.

Cómo se valida: dejando fuera una sesión
========================================
Para cada paciente, cada sesión se puntúa con un modelo entrenado SOLO con sus otras sesiones (leave-one-session-out).
Así el modelo nunca ha visto la grabación que puntúa: no puede reconocerla. Las sesiones (días distintos) son la
unidad natural: los tokens de una misma sesión suelen ir seguidos (tokens_vecinos.py).

Cuándo un paciente "se puede predecir"
======================================
Su AUC (todas sus ventanas puntuadas fuera de su sesión) con un intervalo del 95 % entero por encima de 0,5. El
intervalo se calcula sorteando BLOQUES, no ventanas (las ventanas seguidas se parecen mucho entre sí): cada crisis
es un bloque de ⏳ y cada `BLOQUE_INTERICTAL_S` de un tramo 💤 es un bloque de 💤.

Como con muchos pacientes alguno saldrá "predecible" por suerte, se mide también un CONTROL DE FALSAS CRISIS:
se repite todo con ⏳ falsas (trozos de 💤 elegidos al azar, de la misma duración que el preictal). Los pacientes
que salen "predecibles" con crisis falsas dan la tasa de falsos positivos real del procedimiento.

Crisis líderes
==============
F03.03b vio que casi 2 de cada 3 ⏳ están a pocos minutos de la crisis anterior (racimos): son "recuperándose de
una crisis" tanto como "antes de una crisis". Una crisis líder es la que no tiene otra crisis en los
`LIDER_MIN` minutos anteriores dentro de su grabación (suponiendo que los tokens consecutivos van seguidos). La
primera crisis de una grabación cuenta como líder: no se sabe qué hubo antes de empezar a grabar.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from tfm_comun.entorno.rutas import RUTA_OUTPUTS_FASE_02
from tfm_comun.prediccion.linea_base import MODELOS, crear_modelo
from tfm_comun.prediccion.tokens_vecinos import linea_de_tiempo

RUTA_VENTANAS = RUTA_OUTPUTS_FASE_02 / '03_ventanas' / 'ventanas.csv'
COLUMNAS_VENTANAS = ['clase', 'split', 'patient', 'file', 'tipo', 'grupo', 'crisis_id', 'id_tramo', 'ruta', 'muestra_inicio',
                     'inicio_s', 'fin_s', 'segundos_hasta_crisis', 'en_general', 'rechazada']

MINIMO_CRISIS = 5                  # crisis con ⏳ que necesita un paciente para entrar
MINIMO_SESIONES = 2                # sesiones con ⏳ que necesita (si no, no se puede dejar una fuera)
MINIMO_CRISIS_ENTRENO = 2          # crisis con ⏳ que necesita el entrenamiento de cada ronda (si no, esa sesión no se puntúa)
MAX_INTERICTAL_PACIENTE = 4000     # 💤 por paciente como mucho (4.000 ventanas de 4 s ≈ 4,4 h), repartidas entre sus sesiones
BLOQUE_INTERICTAL_S = 300          # tamaño de los bloques de 💤 al sortear para el intervalo
LIDER_MIN = 30                     # minutos sin crisis antes para que una crisis sea "líder"
NIVEL = 0.95
N_BOOTSTRAP = 500
ESPECIFICIDAD = 0.9


# ----------------------------------------------------------------------------------------- ventanas y pacientes
def leer_ventanas(splits, ruta=RUTA_VENTANAS):
    """Las ventanas de F02.03 que sirven aquí: ⏳ del modelo general, todas las 💤 y las ⚡, sin las rechazadas.

    Devuelve un DataFrame con COLUMNAS_VENTANAS + `sesion` ('aaaaaajy_s003') y `etiqueta` (1 ⏳, 0 💤, -1 ⚡).
    """
    assert ruta.exists(), f'No está {ruta}. Ejecuta antes F02.03 (fase_02/03_ventanas.ipynb).'
    v = pd.read_csv(ruta, usecols=COLUMNAS_VENTANAS, dtype={'crisis_id': str, 'tipo': str, 'grupo': str}, low_memory=False)
    v = v[~v.rechazada.astype(bool) & v.split.isin(splits)]
    v = v[(v.clase == 'interictal') | (v.clase == 'ictal') | ((v.clase == 'preictal') & v.en_general.astype(bool))]
    v = v.drop(columns=['rechazada', 'en_general']).reset_index(drop=True)
    for c in ['crisis_id', 'tipo', 'grupo']:
        v[c] = v[c].fillna('').astype(str).astype(object)       # '' en las 💤 (sin crisis, tipo ni grupo)
    v['sesion'] = v.file.astype(str).str.rsplit('_', n=1).str[0].astype(object)
    v['etiqueta'] = v.clase.map({'preictal': 1, 'interictal': 0, 'ictal': -1}).astype(int)
    return v


def pacientes_candidatos(ventanas, minimo_crisis=MINIMO_CRISIS, minimo_sesiones=MINIMO_SESIONES):
    """Una fila por paciente con ⏳: crisis, sesiones con ⏳, ventanas de cada clase, grupo y si entra.

    Entra si tiene al menos `minimo_crisis` crisis con ⏳ y ⏳ en al menos `minimo_sesiones` sesiones.
    """
    pre = ventanas[ventanas.etiqueta == 1]
    tabla = pre.groupby('patient').agg(split=('split', 'first'), crisis=('crisis_id', 'nunique'),
                                       sesiones_con_preictal=('sesion', 'nunique'), preictal=('etiqueta', 'size'),
                                       grupo=('grupo', lambda g: '/'.join(sorted(set(g) - {''}))),
                                       tipos=('tipo', lambda t: '/'.join(sorted(set(t) - {''}))))
    otras = ventanas[ventanas.etiqueta != 1].groupby(['patient', 'clase']).size().unstack(fill_value=0)
    tabla = tabla.join(otras.rename(columns={'interictal': 'interictal', 'ictal': 'ictal'}), how='left').fillna(0)
    tabla['sesiones'] = ventanas.groupby('patient').sesion.nunique().reindex(tabla.index)
    tabla['entra'] = (tabla.crisis >= minimo_crisis) & (tabla.sesiones_con_preictal >= minimo_sesiones)
    for c in ['crisis', 'sesiones_con_preictal', 'preictal', 'interictal', 'ictal', 'sesiones']:
        if c in tabla:
            tabla[c] = tabla[c].astype(int)
    return tabla.sort_values(['entra', 'crisis'], ascending=False)


def muestrear_interictal(interictal, max_por_paciente=MAX_INTERICTAL_PACIENTE, semilla=42):
    """Máscara (bool, mismo orden) de las 💤 que se usan: como mucho `max_por_paciente` por paciente.

    Se reparten por igual entre las sesiones del paciente (lo que no gasta una sesión pequeña pasa a las demás) y,
    dentro de cada sesión, se toman TRAMOS enteros al azar (el último, cortado: un trozo seguido desde su inicio).
    Trozos seguidos, y no ventanas sueltas, para que el contexto de cada ventana sean sus vecinas (más barato de
    calcular) y para no leer de disco todos los tramos.
    """
    rng = np.random.default_rng(semilla)
    usar = np.zeros(len(interictal), dtype=bool)
    datos = interictal.reset_index(drop=True)
    for _, g in datos.groupby('patient'):
        por_sesion = {s: x for s, x in g.groupby('sesion')}
        restante, sesiones = max_por_paciente, sorted(por_sesion, key=lambda s: len(por_sesion[s]))
        for i, s in enumerate(sesiones):
            cupo = restante // (len(sesiones) - i)
            x = por_sesion[s]
            tramos = list(x.id_tramo.unique())
            rng.shuffle(tramos)
            tomadas = 0
            for t in tramos:
                if tomadas >= cupo:
                    break
                filas_t = x[x.id_tramo == t].sort_values('muestra_inicio')
                elegidas = filas_t.index[:cupo - tomadas]
                usar[elegidas] = True
                tomadas += len(elegidas)
            restante -= tomadas
    return usar


def minutos_desde_crisis_anterior(preictal, edfs, crisis):
    """Para cada fila ⏳: minutos desde el FINAL de la crisis anterior hasta el inicio de la suya (NaN si es la primera de su grabación).

    Usa la línea de tiempo de tokens_vecinos.py (tokens consecutivos seguidos). preictal: filas con file, fin_s y
    segundos_hasta_crisis (el inicio de su crisis = fin_s + segundos_hasta_crisis, en segundos del fichero).
    """
    linea = linea_de_tiempo(edfs)
    c = crisis[['file', 'start', 'stop']].join(linea, on='file', how='inner')
    c['a'] = c.start + c.desfase_s
    c['b'] = c.stop + c.desfase_s
    por_tramo = {t: g.b.to_numpy() for t, g in c.groupby('tramo_continuo')}
    p = preictal[['file', 'fin_s', 'segundos_hasta_crisis']].join(linea, on='file')
    inicio = (p.fin_s + p.segundos_hasta_crisis + p.desfase_s).to_numpy()
    salida = np.full(len(p), np.nan)
    for i, (t, a) in enumerate(zip(p.tramo_continuo.to_numpy(), inicio)):
        fines = por_tramo.get(t)
        if fines is None or np.isnan(a):
            continue
        anteriores = fines[fines < a - 1.0]
        if len(anteriores):
            salida[i] = (a - anteriores.max()) / 60
    return pd.Series(salida, index=preictal.index)


def es_lider(minutos, lider_min=LIDER_MIN):
    """True si la crisis no tiene otra en los `lider_min` minutos anteriores (o es la primera de su grabación)."""
    m = np.asarray(minutos, dtype=float)
    return np.isnan(m) | (m >= lider_min)


def grupos_bootstrap(filas, bloque_s=BLOQUE_INTERICTAL_S):
    """El bloque de cada fila para sortear en el intervalo: su crisis (⏳ y ⚡) o un trozo de `bloque_s` de su tramo (💤)."""
    bloque = (filas.id_tramo.astype(str) + '|' + (filas.inicio_s // bloque_s).astype(int).astype(str)).to_numpy(dtype=object)
    crisis = ('c|' + filas.crisis_id.fillna('').astype(str)).to_numpy(dtype=object)
    return np.where(filas.etiqueta.to_numpy() == 0, bloque, crisis)


def pseudo_preictal(filas, duracion_s, semilla=42):
    """⏳ falsas para el control de falsas crisis: en cada sesión con ⏳, tantos trozos seguidos de 💤 como crisis tiene.

    Parámetros:
        filas: filas de UN paciente (⏳ y 💤, con sesion, crisis_id, id_tramo, muestra_inicio, etiqueta).
        duracion_s: segundos de cada trozo falso (los del preictal: 120 s en el modelo general).
    Devuelve:
        (etiqueta_falsa, crisis_falsa): arrays del mismo largo que filas. etiqueta_falsa = 1 en las 💤 elegidas,
        0 en el resto de 💤 y -2 en las ⏳ verdaderas (que no se usan en el control). crisis_falsa = id del trozo.
    """
    rng = np.random.default_rng(semilla)
    etiqueta = np.where(filas.etiqueta.to_numpy() == 1, -2, 0)
    crisis = np.array([''] * len(filas), dtype=object)
    pos = np.arange(len(filas))
    for s, g in filas.groupby('sesion'):
        n_crisis = g.loc[g.etiqueta == 1, 'crisis_id'].nunique()
        inter = g[g.etiqueta == 0].sort_values(['id_tramo', 'muestra_inicio'])
        if not n_crisis or inter.empty:
            continue
        paso = (inter.inicio_s.diff().where(inter.id_tramo.eq(inter.id_tramo.shift()))).median()
        largo = max(1, int(round(duracion_s / (paso if paso and paso > 0 else 4))))
        candidatos = [(t, x.index.to_numpy()) for t, x in inter.groupby('id_tramo') if len(x) >= largo]
        for k in range(n_crisis):
            if not candidatos:
                break
            t, idx = candidatos[rng.integers(len(candidatos))]
            libres = [i for i in range(len(idx) - largo + 1) if (etiqueta[pos[filas.index.get_indexer(idx[i:i + largo])]] == 0).all()]
            if not libres:
                continue
            i0 = libres[rng.integers(len(libres))]
            elegidas = filas.index.get_indexer(idx[i0:i0 + largo])
            etiqueta[elegidas] = 1
            crisis[elegidas] = f'falsa|{s}|{k}'
    return etiqueta, crisis


# ----------------------------------------------------------------------------------------- entrenar y validar
def entrenar_equilibrado(nombre, hiperparametros, X, y, semilla=42):
    """Un modelo de linea_base.py entrenado con las dos clases pesando lo mismo (cada ventana pesa 1 / ventanas de su clase)."""
    y = np.asarray(y)
    peso = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1)) * len(y)
    modelo = crear_modelo(nombre, hiperparametros, semilla)
    modelo.fit(X, y, **{MODELOS[nombre].CLAVE_PESO: peso})
    return modelo


def pliegues_por_sesion(y, sesiones):
    """El pliegue de cada fila: su sesión si la sesión tiene ⏳; si no, el de una sesión con ⏳ (repartidas por turnos).

    Las sesiones sin ⏳ (solo 💤) no necesitan una ronda propia: se puntúan junto a una sesión con ⏳ y tampoco se
    entrena con ellas en esa ronda. Así hay tantas rondas como sesiones con ⏳ (no 30 en un paciente con 30 días de
    grabación) y el modelo sigue sin haber visto nunca la sesión que puntúa.
    """
    y, sesiones = np.asarray(y), np.asarray(sesiones).astype(object)
    con_pre = sorted(set(sesiones[y == 1]))
    if not con_pre:
        return sesiones
    sin_pre = sorted(set(sesiones) - set(con_pre))
    destino = {s: s for s in con_pre} | {s: con_pre[i % len(con_pre)] for i, s in enumerate(sin_pre)}
    return np.array([destino[s] for s in sesiones], dtype=object)


def validar_dejando_sesion(X, y, sesiones, crisis, nombre, hiperparametros, semilla=42, minimo_crisis_entreno=MINIMO_CRISIS_ENTRENO):
    """Puntuación de cada fila con un modelo que NO ha visto su sesión (leave-one-session-out, dentro de UN paciente).

    Parámetros:
        X, y: características y etiquetas (1 / 0) del paciente.  sesiones, crisis: sesión y crisis de cada fila.
    Devuelve:
        (puntuacion, rondas): puntuación de cada fila (NaN si su sesión no se pudo puntuar: el resto del paciente no
        tenía `minimo_crisis_entreno` crisis o no tenía 💤) y el número de rondas (sesiones con ⏳ puntuadas).
    Las sesiones sin ⏳ se puntúan junto a una sesión con ⏳ (pliegues_por_sesion).
    """
    y, sesiones, crisis = np.asarray(y), np.asarray(sesiones), np.asarray(crisis)
    puntuacion = np.full(len(y), np.nan)
    rondas = 0
    pliegue = pliegues_por_sesion(y, sesiones)
    for s in np.unique(pliegue):
        prueba = pliegue == s
        entreno = ~prueba
        if len(set(crisis[entreno & (y == 1)])) < minimo_crisis_entreno or not (entreno & (y == 0)).any():
            continue
        modelo = entrenar_equilibrado(nombre, hiperparametros, X[entreno], y[entreno], semilla)
        puntuacion[prueba] = modelo.predict_proba(X[prueba])[:, 1]
        rondas += 1
    return puntuacion, rondas


# ----------------------------------------------------------------------------------------- medir
def auroc_por_bloques(y, puntuacion, grupos, n_bootstrap=N_BOOTSTRAP, nivel=NIVEL, semilla=42):
    """AUC y su intervalo sorteando bloques (crisis y trozos de 💤), no ventanas: (auc, inferior, superior). NaN si falta una clase."""
    y, p, g = np.asarray(y), np.asarray(puntuacion, dtype=float), np.asarray(grupos)
    if len(set(y)) < 2:
        return (np.nan, np.nan, np.nan)
    auc = float(roc_auc_score(y, p))
    codigos, unicos = pd.factorize(g)
    clase_bloque = pd.Series(y).groupby(codigos).first().to_numpy()
    rng = np.random.default_rng(semilla)
    muestras = []
    bloques_pos, bloques_neg = np.where(clase_bloque == 1)[0], np.where(clase_bloque == 0)[0]
    for _ in range(n_bootstrap):                     # se sortean por separado los bloques de cada clase
        veces = np.bincount(np.r_[rng.choice(bloques_pos, len(bloques_pos)), rng.choice(bloques_neg, len(bloques_neg))],
                            minlength=len(unicos))
        muestras.append(roc_auc_score(y, p, sample_weight=veces[codigos]))
    inferior, superior = np.percentile(muestras, [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2])
    return auc, float(inferior), float(superior)


def auroc_misma_sesion(y, puntuacion, sesiones):
    """AUC comparando solo ⏳ y 💤 de la MISMA sesión (media pesada por pares). NaN si ninguna sesión tiene las dos clases."""
    t = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(puntuacion, dtype=float), 's': np.asarray(sesiones)})
    aucs, pares = [], []
    for _, g in t.groupby('s'):
        n1, n0 = int((g.y == 1).sum()), int((g.y == 0).sum())
        if n1 and n0:
            aucs.append(roc_auc_score(g.y, g.p))
            pares.append(n1 * n0)
    return float(np.average(aucs, weights=pares)) if aucs else np.nan


def crisis_avisadas(y, puntuacion, crisis, especificidad=ESPECIFICIDAD):
    """(avisadas, total): crisis cuyo ⏳ supera, en mediana, el umbral que deja por debajo el 90 % de las 💤 del paciente.

    En palabras: con un umbral que da la alarma en el 10 % de los ratos tranquilos, ¿en cuántas crisis la mayor
    parte de los 2 minutos de antes estaba por encima?
    """
    y, p, c = np.asarray(y), np.asarray(puntuacion, dtype=float), np.asarray(crisis)
    umbral = np.quantile(p[y == 0], especificidad)
    medianas = pd.Series(p[y == 1]).groupby(c[y == 1]).median()
    return int((medianas > umbral).sum()), int(len(medianas))


def resumen_pacientes(aucs_inferiores, tasa_nula, nivel=NIVEL):
    """Cuántos pacientes salen "predecibles" (intervalo entero por encima de 0,5) y si son más de los que daría la suerte.

    tasa_nula: la proporción de "predecibles" que se espera por azar (la del control de falsas crisis, nunca por
    debajo de (1 - nivel) / 2). Devuelve (predecibles, pacientes, p del test binomial de una cola).
    """
    inferiores = np.asarray(aucs_inferiores, dtype=float)
    inferiores = inferiores[~np.isnan(inferiores)]
    k, n = int((inferiores > 0.5).sum()), int(len(inferiores))
    p0 = max(float(tasa_nula), (1 - nivel) / 2)
    return k, n, (binomtest(k, n, p0, alternative='greater').pvalue if n else np.nan)
