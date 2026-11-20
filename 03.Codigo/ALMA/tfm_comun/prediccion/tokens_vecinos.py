"""
Márgenes de las crisis que están en OTRO token de la misma sesión.

Motivo para cambiar: que cambie cómo se decide si dos tokens de una sesión van seguidos.

TUSZ parte cada sesión en tokens (t000, t001, ...), cada uno su EDF. La cabecera no dice a qué hora
empieza cada uno (la hora está anonimizada a 00.00.00), así que F01.01 mide el lead-in solo dentro de
su fichero, y F02.01 aplica los márgenes (15 min antes de una crisis, margen postictal después) solo
dentro del fichero de la crisis.

Pero los datos dicen que los tokens consecutivos suelen ir SEGUIDOS: duran 300, 601 o 1.800 s (una
grabación larga cortada en trozos) y, cuando una crisis llega al final de un token, el siguiente
empieza con una crisis unas 20 veces más a menudo que un token cualquiera (`evidencia_contiguidad`).
Si van seguidos, una 💤 al final de t003 puede estar a 1 min de una crisis al principio de t004, y una
⏳ al principio de t005 puede ser el postictal de una crisis al final de t004.

Lo prudente para EXCLUIR es suponer que los tokens consecutivos van seguidos (en el peor caso, sin
hueco). No se usa para lo contrario: el lead-in NO se alarga sumando el token anterior.

- `linea_de_tiempo`: segundo de la sesión en que empezaría cada fichero si los consecutivos van seguidos.
- `cerca_de_crisis_de_otro_token`: qué ventanas 💤 o ⏳ caen en los márgenes de una crisis de otro token.
- `evidencia_contiguidad`: las cifras que justifican suponerlo.

Lo usa fase_02/03_ventanas (sección 5): las ventanas marcadas se descartan, como las de amplitud.
"""
import numpy as np
import pandas as pd

from tfm_comun.entorno.rutas import RUTA_OUTPUTS_FASE_01
from tfm_comun.prediccion.umbrales import margen_postictal_s, UMBRALES_LEADIN_COMPARACION_MIN

RUTA_INVENTARIO = RUTA_OUTPUTS_FASE_01 / '01_inventario'
MARGEN_PREVIO_S = max(UMBRALES_LEADIN_COMPARACION_MIN) * 60   # el mismo que usa F02.01 para el 💤 (15 min)
TOLERANCIA_BORDE_S = 2.0                                       # "llega al final" / "empieza al principio" del token


def leer_inventario(ruta=RUTA_INVENTARIO):
    """edfs.csv y crisis.csv de F01.01 (todas las crisis, también las que F02.01 descarta)."""
    edfs = pd.read_csv(ruta / 'edfs.csv')
    crisis = pd.read_csv(ruta / 'crisis.csv')
    return edfs, crisis


def linea_de_tiempo(edfs):
    """Por fichero: `tramo_continuo` (tokens consecutivos de una sesión) y `desfase_s` (segundo en que empezaría).

    Dos ficheros van en el mismo tramo continuo si son de la misma sesión y sus tokens son consecutivos
    (t003 y t004). Si falta un token (t002 y t004), no se sabe cuánto hay entre medias y se corta ahí.
    """
    e = edfs[['patient', 'session', 'token', 'file', 'edf_dur_s']].copy()
    e['n_token'] = e.token.astype(str).str.lstrip('t').astype(int)
    e = e.sort_values(['patient', 'session', 'n_token'])
    sesion = e.patient.astype(str) + '_' + e.session.astype(str)
    nuevo = (sesion != sesion.shift()) | (e.n_token != e.n_token.shift() + 1)
    e['tramo_continuo'] = nuevo.cumsum().astype(int)
    e['desfase_s'] = e.groupby('tramo_continuo').edf_dur_s.cumsum() - e.edf_dur_s
    return e.set_index('file')[['tramo_continuo', 'desfase_s', 'edf_dur_s']]


def _crisis_en_la_linea(crisis, linea):
    c = crisis[['file', 'label', 'start', 'stop']].join(linea, on='file', how='inner')
    c['a_s'] = c.start + c.desfase_s
    c['b_s'] = c.stop + c.desfase_s
    c['margen_s'] = c.label.map(margen_postictal_s).astype(float)
    return c


def cerca_de_crisis_de_otro_token(ventanas, edfs, crisis, margen_previo_s=MARGEN_PREVIO_S):
    """Máscara (array de bool): ventanas 💤 o ⏳ dentro de los márgenes de una crisis de OTRO fichero del tramo continuo.

    ventanas: DataFrame con `file`, `clase`, `inicio_s` y `fin_s` (s desde el inicio de su fichero).
    - 💤 interictal: no puede estar a menos de `margen_previo_s` antes de la crisis ni dentro de la crisis o
      de su margen postictal (lo mismo que exige F02.01 dentro del fichero).
    - ⏳ preictal: no puede caer en una crisis de un token ANTERIOR ni en su margen postictal (lo que F02.01
      descuenta del lead-in dentro del fichero). Una crisis de un token posterior no le afecta.
    Las crisis del mismo fichero no se miran: F02.01 ya las respeta. Las ⚡ ictales nunca se marcan.
    """
    linea = linea_de_tiempo(edfs)
    c = _crisis_en_la_linea(crisis, linea)
    marcada = np.zeros(len(ventanas), dtype=bool)
    v = ventanas[['file', 'clase', 'inicio_s', 'fin_s']].reset_index(drop=True)
    v = v.join(linea[['tramo_continuo', 'desfase_s']], on='file')
    v['pos'] = np.arange(len(v))
    v = v[v.clase.isin(['interictal', 'preictal']) & v.tramo_continuo.isin(set(c.tramo_continuo))]
    crisis_por_tramo = {t: g for t, g in c.groupby('tramo_continuo')}
    for t, g in v.groupby('tramo_continuo'):
        cr = crisis_por_tramo[t]
        t0 = (g.inicio_s + g.desfase_s).to_numpy()[:, None]
        t1 = (g.fin_s + g.desfase_s).to_numpy()[:, None]
        otro = g.file.to_numpy()[:, None] != cr.file.to_numpy()[None, :]
        a, b, m = cr.a_s.to_numpy()[None, :], cr.b_s.to_numpy()[None, :], cr.margen_s.to_numpy()[None, :]
        es_inter = (g.clase == 'interictal').to_numpy()[:, None]
        desfase_crisis = cr.desfase_s.to_numpy()[None, :]
        anterior = desfase_crisis < g.desfase_s.to_numpy()[:, None]           # crisis de un token anterior
        choca_inter = (t1 > a - margen_previo_s) & (t0 < b + m)
        choca_pre = anterior & (t1 > a) & (t0 < b + m)
        malo = otro & np.where(es_inter, choca_inter, choca_pre)
        marcada[g.pos.to_numpy()[malo.any(axis=1)]] = True
    return marcada


def evidencia_contiguidad(edfs, crisis, tolerancia_s=TOLERANCIA_BORDE_S):
    """Cifras que justifican suponer que los tokens consecutivos van seguidos.

    Si fueran grabaciones independientes, que un token empiece con crisis no dependería de cómo acabó el anterior.
    Devuelve un dict con: ficheros, ficheros_con_token_anterior, crisis_al_final (llegan a los últimos
    `tolerancia_s` s de su token), crisis_al_final_con_siguiente, siguiente_empieza_con_crisis, pct_condicionado
    y pct_base (% de todos los tokens que empiezan con crisis).
    """
    linea = linea_de_tiempo(edfs)
    orden = linea.reset_index().sort_values(['tramo_continuo', 'desfase_s'])
    siguiente = dict(zip(orden.file, orden.groupby('tramo_continuo').file.shift(-1)))
    duracion = linea.edf_dur_s
    empieza_con_crisis = set(crisis.loc[crisis.start < tolerancia_s, 'file'])
    al_final = crisis[crisis.stop > crisis.file.map(duracion) - tolerancia_s]
    con_siguiente = [siguiente.get(f) for f in al_final.file if isinstance(siguiente.get(f), str)]
    n_sig_crisis = sum(f in empieza_con_crisis for f in con_siguiente)
    return {
        'ficheros': int(len(linea)),
        'ficheros_con_token_anterior': int(len(linea) - linea.tramo_continuo.nunique()),
        'crisis_al_final': int(len(al_final)),
        'crisis_al_final_con_siguiente': int(len(con_siguiente)),
        'siguiente_empieza_con_crisis': int(n_sig_crisis),
        'pct_condicionado': 100 * n_sig_crisis / len(con_siguiente) if con_siguiente else float('nan'),
        'pct_base': 100 * len(empieza_con_crisis) / len(linea),
    }
