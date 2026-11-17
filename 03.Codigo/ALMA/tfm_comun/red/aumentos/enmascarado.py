"""Enmascarado del espectro (F03.05): tapar a propósito una parte de la entrada para que la red no dependa de ella.

Motivo para cambiar: que cambie qué se tapa o cómo.

Imagen: fase_03/img/aumentos_espectro.png (apagar canales y tapar frecuencias en la entrada de la red)

La red de F03.04 recibe secuencias de espectros ya normalizados por paciente: (lote, ventanas, canales, frecuencias),
donde 0 = "lo normal en este paciente". Dos aumentos que la obligan a no depender de un detalle concreto:
    - apagar_canales: pone a 0 algunos canales en TODA la secuencia (como un electrodo que falla el minuto entero).
      La red no puede apoyarse en un único canal "favorito" de un paciente de train.
    - tapar_frecuencias: pone a 0 una franja de frecuencias seguidas (idea de SpecAugment, Park et al. 2019, para
      espectros de voz). La red no puede apoyarse en una única frecuencia.
Con probabilidad 0 no hacen nada: así la red de F03.04 sigue igual.
"""

import torch


def apagar_canales(x, probabilidad, generador=None):
    """Pone a 0 cada canal de cada fila con esa probabilidad, en todas las ventanas de su secuencia.

    Parámetros:
        x: tensor (lote, ventanas, canales, frecuencias), ya normalizado por paciente (0 = lo normal).
        probabilidad: de cada canal de cada fila, la probabilidad de apagarlo (0,1 = uno de cada 10, de media).
        generador: torch.Generator para que sea repetible.

    Devuelve:
        Un tensor nuevo con la misma forma (o el mismo si probabilidad = 0).
    """
    if not probabilidad:
        return x
    apagar = torch.rand(x.shape[0], 1, x.shape[2], 1, generator=generador) < probabilidad
    return x.masked_fill(apagar.to(x.device), 0.0)


def tapar_frecuencias(x, ancho_maximo, generador=None):
    """Pone a 0, en cada fila, una franja de frecuencias seguidas (de 0 a ancho_maximo), igual en canales y ventanas.

    Parámetros:
        x: tensor (lote, ventanas, canales, frecuencias).  ancho_maximo: frecuencias como mucho (4 = hasta 4 Hz).
        generador: torch.Generator para que sea repetible.

    Devuelve:
        Un tensor nuevo con la misma forma (o el mismo si ancho_maximo = 0).
    """
    if not ancho_maximo:
        return x
    lote, n_frecuencias = x.shape[0], x.shape[-1]
    ancho = torch.randint(0, int(ancho_maximo) + 1, (lote, 1), generator=generador)
    inicio = (torch.rand(lote, 1, generator=generador) * (n_frecuencias - ancho + 1)).long()
    columnas = torch.arange(n_frecuencias).view(1, -1)
    tapar = (columnas >= inicio) & (columnas < inicio + ancho)                    # (lote, frecuencias)
    return x.masked_fill(tapar.view(lote, 1, 1, n_frecuencias).to(x.device), 0.0)
