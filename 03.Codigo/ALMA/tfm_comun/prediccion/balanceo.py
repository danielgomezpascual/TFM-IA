"""
Balanceo: cuántas ventanas 💤 interictales se quedan de cada paciente.

Motivo para cambiar: que cambie la regla para limitar o equilibrar las ventanas interictales.
Las funciones solo deciden QUÉ ventanas se quedan (devuelven máscaras verdadero/falso): no borran
nada ni leen la señal. Así la Fase 3 puede entrenar con el conjunto equilibrado o con todo y pesos.

Imagen: tfm_comun/imagenes/balanceo_por_paciente.png (las dos reglas con 4 pacientes inventados)

Dos reglas, en este orden
=========================
1. Máximo por paciente (todas las particiones). Hay pacientes con muchísimas horas de 💤 interictal
   (en 🟡 dev, uno tiene ~99 h de 338). Sin máximo, ese paciente pesaría más que todos los demás juntos:
   en train el modelo aprendería "su" EEG y en dev/eval la métrica diría sobre todo cómo va con él.
2. Equilibrado por paciente (solo en entrenamiento). Cada paciente aporta EXACTAMENTE `ratio` ventanas
   💤 interictales por cada ⏳ preictal suya (si le faltan 💤, se sortean también sus ⏳). Así:
   - las clases quedan equilibradas en el total (el modelo no gana nada diciendo siempre "💤 interictal");
   - y DENTRO de cada paciente, con lo que el modelo no puede usar al paciente como atajo
     ("las ventanas de este paciente casi siempre son 💤 interictales"). Un paciente sin ⏳ preictales
     no aporta 💤 interictales.
   Al hacerse paciente a paciente, sirve igual cuando se entrena con unos pacientes y se valida con
   otros (validación cruzada por paciente): cada pliegue de entrenamiento sale equilibrado solo.

dev y eval NO se equilibran: tienen que parecerse a la realidad (muchas más 💤 interictales que ⏳ preictales), si no las
métricas (falsas alarmas por hora, precisión) salen infladas.
"""

import numpy as np
import pandas as pd

from tfm_comun.prediccion.ventanas import VENTANA_S, SOLAPE_POR_CLASE

SEMILLA = 42   # misma semilla en todo el TFM: el sorteo de ventanas sale siempre igual


def ventanas_en_horas(horas, ventana_s=VENTANA_S, solape=SOLAPE_POR_CLASE['interictal']):
    """Cuántas ventanas 💤 interictales caben en unas horas de señal.

    Las interictales no se solapan, así que cada una ocupa ventana_s segundos.
    Ejemplo: 2 h -> 2 x 3.600 / 4 = 1.800 ventanas.
    """
    paso_s = ventana_s * (1 - solape)
    return int(round(horas * 3600 / paso_s))


def _sortear(indices_por_paciente, cupo_por_paciente, semilla):
    """Elige al azar `cupo` ventanas de cada paciente (sin repetir). Devuelve el conjunto de índices elegidos."""
    rng = np.random.default_rng(semilla)
    elegidos = []
    for paciente in sorted(indices_por_paciente):       # orden fijo: el resultado no depende del orden de la tabla
        indices = np.sort(np.asarray(indices_por_paciente[paciente]))
        cupo = int(min(len(indices), cupo_por_paciente.get(paciente, 0)))
        if cupo > 0:
            elegidos.extend(rng.choice(indices, size=cupo, replace=False))
    return set(elegidos)


def limitar_por_paciente(ventanas, maximo, semilla=SEMILLA):
    """Máscara de las ventanas 💤 interictales que se quedan con un máximo de `maximo` por paciente.

    Parámetros:
        ventanas: tabla con columnas 'patient' y 'clase' (una fila por ventana).
        maximo: ventanas 💤 interictales como mucho por paciente (ver ventanas_en_horas()).
        semilla: para que el sorteo salga siempre igual.
    Devuelve:
        Serie verdadero/falso con el índice de `ventanas`. Las ⏳ preictales y ⚡ ictales quedan siempre a True; de las
        💤 interictales de un paciente que pasa del máximo, se queda un sorteo de `maximo` repartido por todos sus ficheros.
    Ejemplo: paciente con 5.000 💤 interictales y máximo 1.800 -> se quedan 1.800 al azar; uno con 300 -> se quedan las 300.

    Imagen: tfm_comun/imagenes/balanceo_por_paciente.png
    """
    inter = ventanas[ventanas.clase == 'interictal']
    por_paciente = inter.groupby('patient').groups
    elegidos = _sortear(por_paciente, {p: maximo for p in por_paciente}, semilla)
    return (ventanas.clase != 'interictal') | ventanas.index.isin(list(elegidos))


def equilibrar_por_paciente(ventanas, ratio=1.0, semilla=SEMILLA, exacto=True):
    """Máscara del conjunto de ENTRENAMIENTO equilibrado: cada paciente aporta como mucho
    `ratio` 💤 interictales por cada ⏳ preictal.

    Parámetros:
        ventanas: tabla con 'patient' y 'clase' (solo 💤 interictal y ⏳ preictal, ya sin descartadas y con el máximo aplicado).
        ratio: ventanas 💤 interictales por cada ⏳ preictal dentro de cada paciente (1 = mitad y mitad).
        semilla: para que el sorteo salga siempre igual.
        exacto: si es True (por defecto), cada paciente queda EXACTAMENTE en `ratio`: si tiene menos 💤 interictales
            de las que tocan, se sortean también sus ⏳ preictales para igualarlas, y un paciente sin 💤 interictales
            no entra en el entrenamiento. Si es False, se quedan todas las ⏳ preictales (comportamiento anterior).
    Devuelve:
        Serie verdadero/falso con el índice de `ventanas`.
    Ejemplos (ratio 1), en ventanas preictales / interictales:
        exacto=True:  90 / 1.800 -> 90 / 90 · 90 / 40 -> 40 / 40 · 90 / 0 -> nada · 0 / 500 -> nada.
        exacto=False: 90 / 1.800 -> 90 / 90 · 90 / 40 -> 90 / 40 · 90 / 0 -> 90 / 0 · 0 / 500 -> nada.

    Por qué exacto (lo detectó fase_02/05_auditoria_atajos): si un paciente aporta más ⏳ que 💤, saber de qué
    paciente es una ventana ayuda a adivinar su clase, y cualquier rasgo fijo del paciente (equipo, canales a 0)
    se convierte en una pista. Con el equilibrado exacto, ningún paciente "tira" hacia una clase.

    Imagen: tfm_comun/imagenes/balanceo_por_paciente.png
    """
    pre = ventanas[ventanas.clase == 'preictal']
    inter = ventanas[ventanas.clase == 'interictal']
    n_pre = pre.groupby('patient').size()
    if not exacto:
        cupos = (n_pre * ratio).round().astype(int).to_dict()
        elegidos = _sortear(inter.groupby('patient').groups, cupos, semilla)
        return (ventanas.clase == 'preictal') | ventanas.index.isin(list(elegidos))
    n_inter = inter.groupby('patient').size().reindex(n_pre.index, fill_value=0)
    pre_que_caben = np.minimum(n_pre, np.floor(n_inter / ratio)).astype(int)        # ⏳ que se pueden emparejar
    cupos_pre = pre_que_caben.to_dict()
    cupos_inter = (pre_que_caben * ratio).round().astype(int).to_dict()
    elegidos = _sortear(pre.groupby('patient').groups, cupos_pre, semilla) | \
        _sortear(inter.groupby('patient').groups, cupos_inter, semilla + 1)
    return pd.Series(ventanas.index.isin(list(elegidos)), index=ventanas.index)


def pesos_de_clase(clases):
    """Peso de cada clase para entrenar con TODAS las ventanas en vez de con el conjunto equilibrado.

    Peso = total / (n.º de clases x ventanas de esa clase): la clase rara pesa más y las dos suman lo mismo.
    Ejemplo: 900 💤 interictales y 100 ⏳ preictales -> {'interictal': 0,56, 'preictal': 5,0}.
    Es la alternativa al submuestreo que se puede probar en la Fase 3 (CrossEntropyLoss(weight=...)).
    """
    cuenta = pd.Series(clases).value_counts()
    return {clase: float(len(clases) / (len(cuenta) * n)) for clase, n in cuenta.items()}
