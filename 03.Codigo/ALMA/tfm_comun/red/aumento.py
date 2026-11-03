"""
Aumento de datos: variaciones realistas de una ventana para que la red vea más ejemplos distintos al entrenar.

Motivo para cambiar: que cambie qué aumentos se usan o cómo se hacen.

Imagen: fase_03/img/reflejo_hemisferios.png (el montaje TCP y su espejo izquierda <-> derecha)

Reflejo de hemisferios (Tang et al. 2021, en TUSZ)
==================================================
Se intercambian los canales del lado izquierdo con los del derecho, como si la cabeza se mirara en un espejo.
Una crisis que empieza en el lóbulo temporal izquierdo pasa a "empezar" en el derecho: sigue siendo una
ventana preictal (o interictal) perfectamente creíble. Así la red no aprende "el lado" en vez del patrón.

En el montaje TCP no basta con cambiar unos canales por otros: los de la cadena central cruzan la cabeza
(A1-T3 ... T4-A2) y al reflejarlos quedan al revés. El espejo de A1-T3 es A2-T4 = -(T4-A2): mismo canal,
con el signo cambiado. espejo_tcp() lo calcula solo a partir del montaje (senal/montaje.py).

No se usa el escalado de amplitud (0,8-1,2) del mismo trabajo: con la normalización por ventana se anularía.
"""

import numpy as np
import torch

from tfm_comun.senal.montaje import PARES_TCP


def electrodo_espejo(electrodo):
    """El electrodo del otro lado: los impares son del lado izquierdo y los pares del derecho (sistema 10-20).

    'T3' -> 'T4' · 'FP1' -> 'FP2' · 'A2' -> 'A1'. Los de la línea media ('CZ') se quedan igual.
    """
    letras, numero = electrodo.rstrip('0123456789'), electrodo[len(electrodo.rstrip('0123456789')):]
    if not numero:
        return electrodo
    n = int(numero)
    return f'{letras}{n + 1 if n % 2 else n - 1}'


def espejo_tcp(pares=PARES_TCP):
    """Cómo se refleja el montaje TCP: qué canal va a cada posición y con qué signo.

    Para cada canal (ánodo, cátodo) se reflejan sus dos electrodos y se busca ese par en el montaje:
        - si está tal cual ('FP1-F7' -> 'FP2-F8'), signo +1;
        - si está al revés ('A1-T3' -> 'A2-T4', que en el montaje es 'T4-A2'), signo -1.

    Devuelve:
        (origen, signo): arrays de 22. El canal k de la ventana reflejada es signo[k] * ventana[origen[k]].
    """
    posicion = {par: k for k, par in enumerate(pares)}
    origen, signo = np.zeros(len(pares), dtype=int), np.zeros(len(pares), dtype=np.float32)
    for k, (anodo, catodo) in enumerate(pares):
        espejo = (electrodo_espejo(anodo), electrodo_espejo(catodo))
        if espejo in posicion:
            origen[k], signo[k] = posicion[espejo], 1.0
        else:
            assert espejo[::-1] in posicion, f'El canal {anodo}-{catodo} no tiene espejo en el montaje.'
            origen[k], signo[k] = posicion[espejo[::-1]], -1.0
    return origen, signo


ORIGEN_ESPEJO, SIGNO_ESPEJO = espejo_tcp()


def reflejar_hemisferios(lote, probabilidad=0.5):
    """Refleja al azar algunas ventanas de un lote (izquierda <-> derecha).

    Parámetros:
        lote: tensor (ventanas, 22 canales, muestras).
        probabilidad: de cada ventana, la probabilidad de reflejarla (0,5 = la mitad, al azar).

    Devuelve:
        Un tensor nuevo con la misma forma. El azar es el de torch: sale igual si se fija la semilla.
    """
    elegidas = torch.rand(lote.shape[0]) < probabilidad
    if not elegidas.any():
        return lote
    lote = lote.clone()
    origen = torch.as_tensor(ORIGEN_ESPEJO)
    signo = torch.as_tensor(SIGNO_ESPEJO, dtype=lote.dtype).view(1, -1, 1)
    lote[elegidas] = lote[elegidas][:, origen] * signo
    return lote
