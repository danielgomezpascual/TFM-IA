"""
Utilidades comunes a todos los notebooks del TFM.

Cada módulo tiene una sola responsabilidad (un solo motivo para cambiar):

- particiones.py -> qué particiones tiene el dataset (train, dev, eval).
- iconos.py      -> iconos de etiquetas y particiones en textos y tablas.
- colores.py     -> colores de las particiones en las figuras.
- numeros.py     -> cómo se escribe un número (formato español).
- ejes.py        -> cómo se muestran los ejes de las figuras.

Se importa siempre desde el módulo concreto, para que se vea de dónde viene
cada cosa. Por ejemplo: from tfm_comun.iconos import con_icono
"""
