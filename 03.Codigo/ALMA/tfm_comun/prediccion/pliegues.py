"""
Pliegues: cómo se reparten los pacientes para la validación cruzada por paciente.

Motivo para cambiar: que cambie el número de pliegues o la regla para repartir a los pacientes.

Imagen: tfm_comun/imagenes/pliegues_por_paciente.png (6 pacientes inventados en 3 pliegues)

Para qué
========
Algunos grupos tienen muy pocos pacientes. Las 🟩 ausencias (🔹), por ejemplo, tienen 8 en train,
4 en dev y 1 solo en eval: la cifra de eval dice cómo va con ESE paciente, no si el modelo generaliza.
Por eso, además de eval, se da la cifra de validación cruzada con los 12 pacientes de train + dev:
se entrena con unos pacientes y se valida con los otros, rotando, y se promedia.

Reglas
======
- Un paciente va entero a un solo pliegue (si no, el modelo vería al paciente al entrenar y al validar).
- Los pliegues se llenan para que tengan un número de ventanas ⏳ preictales parecido: se colocan los pacientes de
  más a menos ventanas, cada uno en el pliegue que menos tiene en ese momento. Sin azar: sale siempre igual.
"""

import pandas as pd


def pliegues_por_paciente(ventanas_por_paciente, n_pliegues):
    """Pliegue (0, 1, 2...) de cada paciente, con los pliegues lo más parecidos posible en ventanas.

    Parámetros:
        ventanas_por_paciente: Serie {paciente: n.º de ventanas ⏳ preictales}.
        n_pliegues: pliegues que se quieren. Si hay menos pacientes, se usa uno por paciente
            (validación "deja uno fuera").
    Devuelve:
        Serie {paciente: pliegue}, ordenada por pliegue y paciente.
    Ejemplo: {A: 300, B: 200, C: 120, D: 100} en 2 pliegues -> A al 0; B al 1; C al 1 (200 < 300); D al 0
        (300 < 320) -> pliegue 0 = A + D (400), pliegue 1 = B + C (320).

    Imagen: tfm_comun/imagenes/pliegues_por_paciente.png
    """
    n = max(1, min(n_pliegues, len(ventanas_por_paciente)))
    carga = [0] * n
    pliegue = {}
    orden = sorted(ventanas_por_paciente.items(), key=lambda par: (-par[1], par[0]))
    for paciente, ventanas in orden:
        destino = min(range(n), key=lambda k: (carga[k], k))
        pliegue[paciente] = destino
        carga[destino] += ventanas
    ordenados = sorted(pliegue, key=lambda p: (pliegue[p], p))          # por pliegue y, dentro, por paciente
    serie = pd.Series([pliegue[p] for p in ordenados], index=pd.Index(ordenados, name='patient'), name='pliegue')
    return serie
