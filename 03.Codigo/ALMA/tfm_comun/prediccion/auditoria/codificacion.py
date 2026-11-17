"""Codificación de una pista: convertirla en un número que ordene las ventanas de "más ⏳" a "menos ⏳".

Motivo para cambiar: que cambie cómo se codifican las pistas.

Imagen: fase_02/img/fuerza_por_franjas.png (codificación por franjas)

Regla de oro: la codificación se AJUSTA con 🔵 train y se APLICA igual a 🟡 dev y 🟢 eval (si se usaran sus
etiquetas para codificar, se haría trampa).
"""

import numpy as np
import pandas as pd


# Franjas para la codificación no monótona de las variables numéricas
N_FRANJAS = 10


def codificar_por_tasa(categorias_ajuste, etiqueta_ajuste, categorias):
    """Convierte una variable de categorías (p. ej. el montaje) en un número: la proporción de ⏳ preictal
    que tiene cada categoría en los datos de ajuste (normalmente 🔵 train).

    Para qué: la AUROC necesita un número que ordene las ventanas. Con categorías no hay orden natural;
    así se ordenan de "la categoría más preictal" a "la menos". Se ajusta con train y se aplica igual a
    dev y eval (si no, se haría trampa con sus etiquetas). Una categoría que no estaba en train recibe la
    proporción global de train.
    Ejemplo: en train, 03_tcp_ar_a tiene 60 % de ⏳ preictal y 01_tcp_ar 45 % -> 03_tcp_ar_a = 0,60; 01_tcp_ar = 0,45.
    """
    etiqueta_ajuste = pd.Series(np.asarray(etiqueta_ajuste))
    tasa = etiqueta_ajuste.groupby(np.asarray(categorias_ajuste)).mean()
    return pd.Series(np.asarray(categorias)).map(tasa).fillna(etiqueta_ajuste.mean()).to_numpy()


def codificar_por_franjas(valores_ajuste, etiqueta_ajuste, valores, n_franjas=N_FRANJAS):
    """Convierte una variable numérica en "proporción de ⏳ preictal de su franja", ajustada con los datos de ajuste.

    Para qué: la AUROC de la variable tal cual solo ve relaciones de "más / menos". Si la pista tiene forma
    (p. ej. el minuto 0-1 es casi solo 💤, los minutos 2-5 casi solo ⏳ y el resto mezclado), la AUROC tal cual
    sale cerca de 0,5 aunque un modelo sí podría usarla. Con esta codificación también se capta.
    Qué hace:
        1. Si la variable tiene n_franjas valores distintos o menos (p. ej. 0/1, o 0/2 canales a cero), cada valor es
           su propia franja. Si no, se corta en n_franjas franjas con el mismo número de ventanas (cuantiles de los
           datos de ajuste; con valores muy repetidos salen menos).
        2. Cada franja vale la proporción de ⏳ preictal que tiene en los datos de ajuste.
        3. Se aplica igual a los datos que se quieren medir (dev y eval no usan sus etiquetas).
    Devuelve: array con un número por ventana de `valores`.
    """
    ajuste = np.asarray(valores_ajuste, dtype=float)
    if len(np.unique(ajuste)) <= n_franjas:                                       # pocos valores: uno por franja
        return codificar_por_tasa(ajuste, etiqueta_ajuste, np.asarray(valores, dtype=float))
    interiores = np.unique(np.quantile(ajuste, np.linspace(0, 1, n_franjas + 1)[1:-1]))   # bordes entre franjas
    franja = lambda x: np.searchsorted(interiores, np.asarray(x, dtype=float), side='right')
    return codificar_por_tasa(franja(ajuste), etiqueta_ajuste, franja(valores))
