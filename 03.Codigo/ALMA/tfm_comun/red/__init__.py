"""
Red: todo lo que tiene que ver con PyTorch (cómo le llegan los datos a la red, las redes y su entrenamiento).

Ficheros sueltos: paquetes.py, reproducibilidad.py (F03.01) · modelo.py (catálogo de redes) · preparacion.py (F03.05) ·
                  grafo.py (F03.08)
Carpetas (cada fichero con una sola responsabilidad):
    ventanas/        F03.01: Dataset, preparación del lote y DataLoader de ventanas sueltas.
    entradas/        qué números recibe la red de cada ventana.
    aumentos/        aumento de datos (reflejo de hemisferios, enmascarado del espectro).
    lotes/           secuencias de espectros y orden de las filas en cada época.
    entreno/         puntos de control, puntuar, parada temprana, bucle de entrenamiento y entrenador de variantes.
    experimentos/    huella y registro de experimentos (terminados y parciales).
    autosupervisado/ F03.07: corpus sin etiquetas y preentrenamiento.
    redes/           cada red y cada pieza de red en su fichero.
"""
