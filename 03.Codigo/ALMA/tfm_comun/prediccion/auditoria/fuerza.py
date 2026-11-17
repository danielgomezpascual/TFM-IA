"""Fuerza de una pista: cuánto separa las ⏳ preictales de las 💤 interictales, de 0,5 (nada) a 1 (todo).

Motivo para cambiar: que cambie cómo se mide la fuerza o los cortes del semáforo.

Imagen: fase_02/img/auroc_y_fuerza.png (la AUROC, la fuerza y la dirección)

Como una pista puede separar "al revés" (p. ej. las ⏳ salen ANTES en el fichero, AUROC < 0,5), se usa la
FUERZA = max(AUROC, 1 - AUROC) y, aparte, la dirección.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


# ── Semáforo de la fuerza ──────────────────────────────────────────────────────────────────────────
# No hay un umbral "oficial": son cortes habituales para leer una AUROC. Por debajo de 0,55 la variable
# casi no separa; por encima de 0,65 separa lo bastante como para que una red la aproveche.
FUERZA_DEBIL = 0.55


FUERZA_FUERTE = 0.65


def fuerza_auroc(etiqueta, valores):
    """AUROC de una variable, su fuerza y su dirección.

    Parámetros:
        etiqueta: 1 = ⏳ preictal, 0 = 💤 interictal.  valores: la variable (un número por ventana).
    Devuelve:
        (auroc, fuerza, direccion):
            auroc: probabilidad de que una ⏳ preictal al azar tenga un valor MAYOR que una 💤 interictal al azar.
            fuerza: max(auroc, 1 - auroc), de 0,5 (no separa) a 1 (separa todo).
            direccion: 'más alta en ⏳ preictal', 'más baja en ⏳ preictal' o 'igual'.
        Si solo hay una clase o la variable es constante, (0,5, 0,5, 'igual').
    Ejemplo: minuto_en_fichero con AUROC 0,30 -> fuerza 0,70, 'más baja en ⏳ preictal' (salen antes en el fichero).
    """
    etiqueta, valores = np.asarray(etiqueta), np.asarray(valores, dtype=float)
    if len(np.unique(etiqueta)) < 2 or np.nanstd(valores) == 0:
        return 0.5, 0.5, 'igual'
    auroc = float(roc_auc_score(etiqueta, valores))
    direccion = 'igual' if abs(auroc - 0.5) < 0.005 else ('más alta en ⏳ preictal' if auroc > 0.5 else 'más baja en ⏳ preictal')
    return auroc, max(auroc, 1 - auroc), direccion


def fuerza_dentro_del_paciente(ventanas, variable, columna_paciente='patient', columna_etiqueta='etiqueta'):
    """Fuerza de una variable DENTRO de cada paciente, promediada (pesando por ventanas).

    Por qué importa: el entrenamiento está equilibrado paciente a paciente (balanceo.py), así que una
    pista que solo cambia ENTRE pacientes (p. ej. "este paciente se grabó con un equipo de 250 Hz") no
    le sirve a la red: cada paciente aporta mitad y mitad. La pista peligrosa es la que cambia DENTRO
    del paciente: sus ⏳ preictales y sus 💤 interictales salen de situaciones distintas.
    Para cada paciente con las dos clases se calcula la AUROC; se promedian y se devuelve la fuerza de ese
    promedio. Un paciente con la variable constante cuenta como 0,5 (dentro de él, no separa).

    Devuelve:
        (auroc_media, fuerza, n_pacientes).
    """
    aurocs, pesos = [], []
    for _, v in ventanas.groupby(columna_paciente):
        if v[columna_etiqueta].nunique() < 2:
            continue
        aurocs.append(fuerza_auroc(v[columna_etiqueta], v[variable])[0])
        pesos.append(len(v))
    if not aurocs:
        return 0.5, 0.5, 0
    media = float(np.average(aurocs, weights=pesos))
    return media, max(media, 1 - media), len(aurocs)


def semaforo(fuerza):
    """🟢 / 🟡 / 🔴 según la fuerza de una variable: < FUERZA_DEBIL (0,55) no separa, hasta FUERZA_FUERTE (0,65)
    separa un poco, por encima separa lo bastante como para que una red la use."""
    if fuerza < FUERZA_DEBIL:
        return '🟢 no separa'
    if fuerza < FUERZA_FUERTE:
        return '🟡 separa un poco'
    return '🔴 separa mucho'
