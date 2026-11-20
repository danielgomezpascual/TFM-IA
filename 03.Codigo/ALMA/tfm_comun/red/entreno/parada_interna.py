"""Parada interna: elegir la mejor época con pacientes apartados de 🔵 train, no con 🟡 dev.

Motivo para cambiar: que cambie con qué filas se decide cuándo parar y qué época se guarda.

Imagen: fase_03/img/parada_temprana.png (las curvas de aprendizaje y el momento de parar)

El problema (visto en F03.04, 2026-11-19)
=========================================
Con la parada temprana sobre 🟡 dev, la red se queda con la época en la que la AUC de dev es MÁXIMA. Si esa curva
es ruidosa (en F03.04 saltaba ±0,04 de una época a otra sin que la pérdida de dev bajase), el máximo es en buena
parte suerte, y la AUC que luego se publica en dev es la de ese máximo: optimista. Además, con esa misma AUC se
compara la red con XGBoost.

La solución
===========
Se aparta una parte de los PACIENTES de 🔵 train (por defecto, un 20 %) solo para decidir cuándo parar. La red
aprende con el resto, y 🟡 dev ya no influye en nada del entrenamiento: su AUC es una medida limpia.
Los pacientes apartados se eligen entre los que tienen ⏳ y 💤 (si no, la AUC de parada no se puede calcular), con
una semilla fija: todas las variantes y semillas paran con los MISMOS pacientes, y se pueden comparar.

El precio: la red aprende con menos pacientes. Con unos 80 en train, apartar 16 se nota algo; es el coste de no
elegir la época mirando la nota que se publica.
"""

import numpy as np


def separar_parada(posiciones, pacientes, etiquetas, fraccion=0.2, semilla=42, minimo=4):
    """Parte las filas de entrenamiento en (para aprender, para decidir cuándo parar), por pacientes.

    Parámetros:
        posiciones: filas de entrenamiento (posiciones en la tabla de filas).
        pacientes, etiquetas: paciente y etiqueta (1 = ⏳, 0 = 💤) de CADA FILA DE LA TABLA (se indexan con posiciones).
        fraccion: parte de los pacientes con las dos clases que se apartan para la parada (0 = no se aparta nada).
        semilla: fija qué pacientes se apartan.
        minimo: pacientes mínimos para la parada; si no hay tantos con las dos clases, no se aparta nada.

    Devuelve:
        (aprender, parar, pacientes_parada): posiciones para entrenar, posiciones para la parada temprana y la lista
        de pacientes apartados. Si no se aparta nada, parar está vacío (y hay que parar con otra cosa).
    """
    posiciones = np.asarray(posiciones)
    pacientes_e = np.asarray(pacientes)[posiciones]
    etiquetas_e = np.asarray(etiquetas)[posiciones]
    con_las_dos = sorted(p for p in np.unique(pacientes_e) if len(np.unique(etiquetas_e[pacientes_e == p])) == 2)
    n = int(round(fraccion * len(con_las_dos)))
    if fraccion <= 0 or n < minimo:
        return posiciones, posiciones[:0], []
    apartados = sorted(np.random.default_rng(semilla).choice(con_las_dos, size=n, replace=False).tolist())
    es_parada = np.isin(pacientes_e, apartados)
    return posiciones[~es_parada], posiciones[es_parada], apartados
