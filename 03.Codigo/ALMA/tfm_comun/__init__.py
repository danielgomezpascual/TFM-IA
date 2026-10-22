"""
Utilidades comunes a todos los notebooks del TFM.

Cada módulo tiene una sola responsabilidad (un solo motivo para cambiar):

- rutas.py       -> dónde están el dataset y las salidas.
- particiones.py -> qué particiones tiene el dataset (train, dev, eval).
- iconos.py      -> iconos de etiquetas, particiones, tipos de canal y regiones.
- colores.py     -> colores de las figuras (particiones, etiquetas, periodos, canales, regiones, bandas).
- numeros.py     -> cómo se escribe un número (formato español).
- ejes.py        -> cómo se muestran los ejes de las figuras.
- canales.py     -> qué es cada canal del EDF (tipo, región, referencia).
- edf.py         -> lectura de un EDF de TUSZ con MNE.
- anotaciones.py -> lectura de las anotaciones .csv/.csv_bi y crisis de un fichero.
- montaje.py     -> montaje bipolar TCP de 22 canales.
- bandas.py      -> bandas de frecuencia del EEG (delta, theta, alfa, beta, gamma).
- cabeza.py      -> dibujo de la cabeza con los electrodos 10-20 y el montaje.

Se importa siempre desde el módulo concreto, para que se vea de dónde viene
cada cosa. Por ejemplo: from tfm_comun.iconos import con_icono
"""
