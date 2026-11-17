"""Aumentos de datos: variaciones realistas de un ejemplo para que la red vea más casos distintos al entrenar.

    reflejo_hemisferios.py -> izquierda <-> derecha del montaje TCP (Tang et al.). Sirve para la señal y para el espectro.
    enmascarado.py         -> apagar canales y tapar frecuencias en el espectro normalizado (F03.05).

Todos se aplican SOLO al entrenar y con un generador con semilla: con la misma semilla salen igual.
"""



