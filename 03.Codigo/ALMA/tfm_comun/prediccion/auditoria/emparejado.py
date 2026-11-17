"""Emparejado por posición: un conjunto de CONTROL en el que las 💤 salen de los mismos minutos del fichero que las ⏳.

Motivo para cambiar: que cambie cómo se emparejan las ventanas.

Imagen: fase_02/img/emparejado_por_posicion.png
"""

import numpy as np


def emparejar_por_posicion(ventanas, ancho_min=1.0, ratio=1.0, semilla=42,
                           columna_paciente='patient', columna_minuto='minuto_en_fichero'):
    """Máscara de un conjunto de CONTROL en el que las 💤 interictales salen de los mismos minutos del
    fichero que las ⏳ preictales, paciente a paciente.

    Qué hace:
        1. Parte el fichero en franjas de `ancho_min` minutos (0-1, 1-2, 2-3...).
        2. Para cada paciente y franja, cuenta sus ⏳ preictales (n) y sortea round(ratio x n) 💤 interictales
           SUYAS de esa misma franja (todas las que haya, si tiene menos).
        3. Se quedan todas las ⏳ preictales y las 💤 interictales sorteadas. Los pacientes sin ⏳ preictales no aportan.
        Así, si la red acierta solo porque "las ⏳ preictales están al principio del fichero", en este conjunto
        ya no le sirve: las dos clases tienen la misma posición. Si su nota baja mucho aquí, usaba el atajo.

    Parámetros:
        ventanas: tabla con 'clase', paciente y minuto_en_fichero (de metadatos_de_ventanas).
        ancho_min: anchura de las franjas (min). Más estrecha = emparejado más fino, pero menos ventanas.
        ratio: 💤 interictales por cada ⏳ preictal en cada franja.
        semilla: para que el sorteo salga siempre igual.

    Devuelve:
        (mascara, cumplido): Serie verdadero/falso con el índice de `ventanas`, y la fracción de 💤 interictales
        pedidas que se han podido conseguir (1 = todas; menos de 1 = en algunas franjas faltaban).
    Ejemplo: un paciente con 30 ⏳ preictales entre los minutos 3 y 4 y 200 💤 interictales en esa franja ->
        se sortean 30 de esas 200 (con ratio 1).
    """
    rng = np.random.default_rng(semilla)
    franja = np.floor(ventanas[columna_minuto] / ancho_min).astype(int)
    clave = [ventanas[columna_paciente], franja]
    n_pre = ventanas.clase.eq('preictal').groupby(clave).sum()                     # ⏳ preictales por (paciente, franja)
    candidatas = ventanas.index[ventanas.clase == 'interictal'].to_series() \
        .groupby([c[ventanas.clase == 'interictal'] for c in clave]).apply(lambda s: np.sort(s.to_numpy()))
    elegidas, pedidas, conseguidas = [], 0, 0
    for (paciente, f), n in n_pre[n_pre > 0].sort_index().items():               # orden fijo: el sorteo no depende del orden de la tabla
        disponibles = candidatas.get((paciente, f), np.array([], dtype=int))
        objetivo = int(round(n * ratio))
        k = min(len(disponibles), objetivo)
        pedidas += objetivo
        conseguidas += k
        if k:
            elegidas.extend(rng.choice(disponibles, size=k, replace=False))
    mascara = (ventanas.clase == 'preictal') | ventanas.index.isin(elegidas)
    return mascara, (conseguidas / pedidas if pedidas else np.nan)
