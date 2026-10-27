"""
Utilidades comunes a todos los notebooks del TFM.

Cada módulo tiene una sola responsabilidad (un solo motivo para cambiar):

- rutas.py       -> dónde están el dataset y las salidas.
- particiones.py -> qué particiones tiene el dataset (train, dev, eval).
- iconos.py      -> iconos de etiquetas, grupos, particiones, tipos de canal y regiones.
- colores.py     -> colores de las figuras (particiones, etiquetas, periodos, canales, regiones, bandas).
- numeros.py     -> cómo se escribe un número o una duración (formato español).
- ejes.py        -> cómo se muestran los ejes de las figuras.
- canales.py     -> qué es cada canal del EDF (tipo, región, referencia).
- edf.py         -> lectura de un EDF de TUSZ con MNE.
- anotaciones.py -> lectura de las anotaciones .csv/.csv_bi y crisis de un fichero.
- montaje.py     -> montaje bipolar TCP de 22 canales.
- bandas.py      -> bandas de frecuencia del EEG (delta, theta, alfa, beta, gamma).
- cabeza.py      -> dibujo de la cabeza con los electrodos 10-20 y el montaje.
- tipos_crisis.py -> cada tipo de crisis: ficha clínica y grupo de inicio (ILAE).
- umbrales.py    -> umbrales de predicción: lead-in, SPH y margen postictal de cada tipo.
- versiones.py   -> versiones de Python y de las librerías de cada notebook.
- tablas.py      -> cómo se muestran las tablas (separadores entre particiones, formato).
- esquemas.py    -> esquema de los periodos alrededor de una crisis (preictal, SPH, postictal...).

Se importa siempre desde el módulo concreto, para que se vea de dónde viene
cada cosa. Por ejemplo: from tfm_comun.iconos import con_icono
"""
