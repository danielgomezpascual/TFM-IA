"""AUROC por paciente: la nota mirando a cada paciente por separado.

Motivo para cambiar: que cambie cómo se calcula la nota de cada paciente o la nota dentro del paciente.

Imagen: fase_03/img/auc_dentro_paciente.png (la AUC global frente a la AUC dentro del paciente)

Por qué importa
===============
La AUROC global mezcla pacientes. Un modelo puede sacar buena nota global porque distingue PACIENTES (unos dan
puntuaciones altas a todo, otros bajas) y no las ventanas de cada uno. Una app vigila siempre a la misma persona:
lo que necesita es ordenar bien los momentos de ESE paciente, que es lo que miden estas funciones.
    auroc_por_paciente      -> una AUROC por paciente (tabla).
    auroc_dentro_paciente   -> un solo número: la media de esas AUROC, pesando por los pares ⏳-💤 de cada paciente.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tfm_comun.prediccion.evaluacion.bootstrap import N_BOOTSTRAP, NIVEL


def auroc_por_paciente(etiqueta, puntuacion, pacientes):
    """AUROC de cada paciente por separado: ¿el modelo ordena bien las ventanas DENTRO de cada paciente?

    Qué hace:
        Para cada paciente con ventanas de las dos clases, la AUROC solo con sus ventanas. Los pacientes con una
        sola clase (p. ej. sin crisis: solo 💤) no tienen AUROC propia y no salen.

    Por qué importa: la AUROC global mezcla pacientes. Un modelo puede sacar buena nota global porque distingue
    PACIENTES (unos dan puntuaciones altas a todo, otros bajas) y no las ventanas de cada uno. La AUROC por
    paciente mide solo lo segundo, que es lo que haría falta en una app para un paciente concreto.

    Devuelve:
        Tabla con una fila por paciente: auroc, ventanas ⏳ y 💤.
    """
    tabla = pd.DataFrame({'etiqueta': np.asarray(etiqueta), 'puntuacion': np.asarray(puntuacion, dtype=float),
                          'paciente': np.asarray(pacientes)})
    filas = {}
    for paciente, t in tabla.groupby('paciente'):
        if t.etiqueta.nunique() == 2:
            filas[paciente] = {'auroc': roc_auc_score(t.etiqueta, t.puntuacion),
                               '⏳ preictal': int(t.etiqueta.sum()), '💤 interictal': int((1 - t.etiqueta).sum())}
    resultado = pd.DataFrame(filas).T
    resultado.index.name = 'paciente'
    return resultado.astype({'⏳ preictal': int, '💤 interictal': int}) if len(resultado) else resultado


def _auc_y_pares_por_paciente(etiqueta, puntuacion, pacientes):
    """(pacientes, AUC de cada uno, pares ⏳-💤 de cada uno), solo de los pacientes con las dos clases."""
    tabla = pd.DataFrame({'e': np.asarray(etiqueta), 'p': np.asarray(puntuacion, dtype=float), 'paciente': np.asarray(pacientes)})
    ids, aucs, pares = [], [], []
    for paciente, t in tabla.groupby('paciente', sort=True):
        n_pos = int(t.e.sum())
        n_neg = len(t) - n_pos
        if n_pos and n_neg:
            ids.append(paciente)
            aucs.append(roc_auc_score(t.e, t.p))
            pares.append(n_pos * n_neg)
    return np.asarray(ids), np.asarray(aucs, dtype=float), np.asarray(pares, dtype=float)


def auroc_dentro_paciente(etiqueta, puntuacion, pacientes):
    """AUROC DENTRO DEL PACIENTE: se coge un paciente, una ⏳ y una 💤 SUYAS, y se mira si la ⏳ sale más alta.

    Qué hace:
        La AUROC de cada paciente (auroc_por_paciente) y su media ponderada por el número de pares ⏳-💤 de ese
        paciente. Es lo mismo que la AUROC global pero comparando solo ventanas del MISMO paciente.

    Por qué importa (F03.05): la AUROC global también premia dar puntuaciones distintas a pacientes distintos (a unos
    todo alto, a otros todo bajo). Una app vigila siempre a la misma persona: lo que necesita es ordenar bien los
    momentos de ESE paciente, que es justo lo que mide esta nota.

    Devuelve:
        Un número de 0 a 1 (0,5 = azar). NaN si ningún paciente tiene las dos clases.
    """
    _, aucs, pares = _auc_y_pares_por_paciente(etiqueta, puntuacion, pacientes)
    return float((aucs * pares).sum() / pares.sum()) if len(aucs) else float('nan')


def auroc_dentro_paciente_con_intervalo(etiqueta, puntuacion, pacientes, n_bootstrap=N_BOOTSTRAP, nivel=NIVEL, semilla=42):
    """auroc_dentro_paciente() y su intervalo con bootstrap por paciente: (auroc, inferior, superior).

    Se sortean, con repetición, los pacientes que tienen las dos clases (los demás no cuentan en esta nota).
    """
    _, aucs, pares = _auc_y_pares_por_paciente(etiqueta, puntuacion, pacientes)
    valor = float((aucs * pares).sum() / pares.sum())
    rng = np.random.default_rng(semilla)
    muestras = [(aucs[i] * pares[i]).sum() / pares[i].sum() for i in (rng.integers(0, len(aucs), len(aucs)) for _ in range(n_bootstrap))]
    inferior, superior = np.percentile(muestras, [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2])
    return valor, float(inferior), float(superior)


def diferencia_auroc_dentro_paciente_con_intervalo(etiqueta, puntuacion_a, puntuacion_b, pacientes, n_bootstrap=N_BOOTSTRAP, nivel=NIVEL, semilla=42):
    """Diferencia (A - B) de la AUROC dentro del paciente, con los mismos sorteos de pacientes: (diferencia, inferior, superior)."""
    _, a, pares = _auc_y_pares_por_paciente(etiqueta, puntuacion_a, pacientes)
    _, b, _ = _auc_y_pares_por_paciente(etiqueta, puntuacion_b, pacientes)
    diferencia = float(((a - b) * pares).sum() / pares.sum())
    rng = np.random.default_rng(semilla)
    muestras = [((a[i] - b[i]) * pares[i]).sum() / pares[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(n_bootstrap))]
    inferior, superior = np.percentile(muestras, [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2])
    return diferencia, float(inferior), float(superior)
