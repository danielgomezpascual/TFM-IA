"""
Cuánto crece la potencia de cada electrodo al inicio de una crisis respecto a la señal basal (en dB).

Motivo para cambiar: que cambie cómo se mide el aumento electrodo a electrodo.
Es el paso 4 de origen_crisis() (fuentes.py) por separado y sin el cerebro plantilla: se calcula en
menos de un segundo, así que el visor lo enseña al momento, antes del cálculo lento con sLORETA.
Mismo filtrado (BANDA_FUENTES_HZ + notch) y misma referencia (promedio) que origen_crisis().
"""

import numpy as np
import pandas as pd

from tfm_comun.senal.calidad import UMBRAL_PLANO_UV
from tfm_comun.senal.filtrado import NOTCH_HZ
from tfm_comun.senal.fuentes import BANDA_FUENTES_HZ, electrodos_con_posicion
from tfm_comun.visual.inicio_crisis import aumento_por_canal


def electrodos_planos(raw, tramo):
    """Electrodos con la señal plana (desviación < UMBRAL_PLANO_UV) en un tramo: sueltos o apagados.
    Hay que quitarlos: su potencia basal es casi 0 y el aumento saldría disparado."""
    inicio, fin = raw.time_as_index(list(tramo))
    desviacion = raw.get_data(picks='eeg')[:, inicio:fin].std(axis=1) * 1e6
    return [c for c, d in zip(raw.copy().pick('eeg').ch_names, desviacion) if d < UMBRAL_PLANO_UV]


def aumento_por_electrodo(raw, tramo_basal, tramo_inicio, excluir=()):
    """Aumento de potencia (dB) de cada electrodo en el tramo de inicio respecto al basal.

    Qué hace:
        1. Electrodos 10-20 con posición (sin orejas ni los de excluir), como en origen_crisis().
        2. Notch + paso banda BANDA_FUENTES_HZ y referencia promedio (cada electrodo menos la media de todos:
           así la referencia del equipo no favorece a ningún electrodo).
        3. Potencia media en cada tramo y cociente en dB con aumento_por_canal() (inicio_crisis.py).
           +10 dB = 10 veces más potencia que en el basal; 0 dB = sin cambio.

    Parámetros:
        raw: Raw REFERENCIAL (de leer_edf_tusz, sin montaje TCP). Los tramos, en segundos de raw.times.
        excluir: electrodos que no se usan (p. ej. los de electrodos_planos()).

    Devuelve:
        pd.Series {electrodo: aumento en dB}, de más a menos aumento.
    """
    eeg = electrodos_con_posicion(raw, excluir)
    eeg.notch_filter(NOTCH_HZ, verbose='error').filter(*BANDA_FUENTES_HZ, verbose='error')
    eeg.set_eeg_reference('average', verbose='error')
    aumento = aumento_por_canal(eeg.get_data() * 1e6, eeg.times, eeg.ch_names, tramo_basal, tramo_inicio)
    return pd.Series(aumento, name='aumento (dB)').replace([np.inf, -np.inf], np.nan).dropna() \
        .sort_values(ascending=False)
