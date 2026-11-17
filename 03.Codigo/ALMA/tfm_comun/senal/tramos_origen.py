"""
Qué dos tramos se comparan para ver DÓNDE se origina una crisis: su inicio frente a una señal basal.

Motivo para cambiar: que cambie cómo se eligen los tramos (duraciones, márgenes, qué hacer si no hay basal).
Mismos criterios que F01.03v (sección del origen): los primeros VENTANA_INICIO_S de la crisis frente a
hasta BASAL_MAXIMO_S de señal sin crisis de antes. Aquí el basal acaba MARGEN_ANTES_S antes de la crisis
(en F01.03v, 1 min antes del preictal), porque en el visor muchas crisis tienen poco tiempo libre delante.
Lo usa el visor de EEG; el cálculo en sí está en fuentes.py (origen_crisis) e inicio_crisis.py.
"""

import numpy as np

from tfm_comun.senal.calidad import UMBRAL_PLANO_UV

VENTANA_INICIO_S = 5          # primeros segundos de la crisis que se comparan
BASAL_MAXIMO_S = 120          # señal basal: como mucho 2 min
BASAL_MINIMO_S = 20           # con menos, la potencia basal se estima mal
MARGEN_ANTES_S = 10           # el basal acaba 10 s antes de la crisis (para no meter su arranque)
MARGEN_TRAS_CRISIS_S = 60     # tras otra crisis se deja 1 min (postictal: señal alterada)


def tramos_origen(crisis, indice, duracion_s):
    """Elige el tramo de inicio de una crisis y el tramo basal con el que se compara.

    Qué hace:
        1. Inicio: los primeros VENTANA_INICIO_S de la crisis (o toda, si dura menos).
        2. Basal: el tramo sin crisis justo ANTES, acabando MARGEN_ANTES_S antes del inicio y empezando
           MARGEN_TRAS_CRISIS_S después de la crisis anterior (o al principio del fichero).
        3. Si antes no hay al menos BASAL_MINIMO_S, usa el hueco sin crisis más largo del fichero
           (con los mismos márgenes) y lo dice en la nota.

    Parámetros:
        crisis: tabla de crisis_del_fichero() (inicio_s, fin_s, ordenada por inicio).
        indice: posición de la crisis en esa tabla.
        duracion_s: duración del fichero.

    Devuelve:
        (tramo_inicio, tramo_basal, nota): tramos como (inicio, fin) en segundos del fichero.
        tramo_basal es None si no hay ningún hueco de BASAL_MINIMO_S en todo el fichero.
    """
    t_crisis = crisis.inicio_s.iloc[indice]
    tramo_inicio = (t_crisis, t_crisis + min(VENTANA_INICIO_S, crisis.duracion_s.iloc[indice]))

    huecos = _huecos_sin_crisis(crisis, duracion_s)
    antes = [h for h in huecos if h[1] <= t_crisis - MARGEN_ANTES_S + 1e-6]
    if antes and antes[-1][1] - antes[-1][0] >= BASAL_MINIMO_S:
        desde, hasta = antes[-1]
        return tramo_inicio, (max(desde, hasta - BASAL_MAXIMO_S), hasta), 'basal justo antes de la crisis'

    validos = [h for h in huecos if h[1] - h[0] >= BASAL_MINIMO_S]
    if not validos:
        return tramo_inicio, None, (f'no hay {BASAL_MINIMO_S} s seguidos sin crisis en el fichero '
                                    'para compararlos con el inicio')
    desde, hasta = max(validos, key=lambda h: h[1] - h[0])
    return tramo_inicio, (desde, min(hasta, desde + BASAL_MAXIMO_S)), \
        f'antes de la crisis no hay {BASAL_MINIMO_S} s libres: basal de otro tramo sin crisis del fichero'


def _huecos_sin_crisis(crisis, duracion_s):
    """Tramos del fichero sin crisis, con los márgenes de antes y después de cada crisis ya quitados."""
    huecos, desde = [], 0.0
    for c in crisis.sort_values('inicio_s').itertuples():
        hasta = c.inicio_s - MARGEN_ANTES_S
        if hasta > desde:
            huecos.append((desde, hasta))
        desde = max(desde, c.fin_s + MARGEN_TRAS_CRISIS_S)
    if duracion_s > desde:
        huecos.append((desde, duracion_s))
    return huecos


def recortar_inicio_plano(datos_uv, fs, tramo, leido_desde_s):
    """Retrasa el comienzo de un tramo hasta que la señal deja de estar plana.

    Muchos EDF de TUSZ empiezan con unos segundos a 0 (el equipo aún no grababa). Si el basal cae ahí,
    su potencia es casi 0 y el aumento sale infinito. Se busca el primer segundo del tramo en que la
    desviación típica mediana de los electrodos supera UMBRAL_PLANO_UV (calidad.py).

    Parámetros:
        datos_uv: matriz (electrodos x muestras) en µV, leída desde leido_desde_s.
        fs: Hz. tramo: (inicio, fin) en segundos del fichero.

    Devuelve:
        El tramo con el inicio retrasado (el fin no cambia).
    """
    for segundo in np.arange(tramo[0], tramo[1] - 1, 1.0):
        desde = int((segundo - leido_desde_s) * fs)
        if np.median(datos_uv[:, desde:desde + int(fs)].std(axis=1)) > UMBRAL_PLANO_UV:
            return (float(segundo), tramo[1])
    return (tramo[1], tramo[1])
