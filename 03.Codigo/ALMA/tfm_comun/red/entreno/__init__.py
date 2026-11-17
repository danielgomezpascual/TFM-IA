"""Entreno: cómo aprende una red de predicción y cómo se repite el entrenamiento para varias variantes.

Imagen: fase_03/img/parada_temprana.png (las curvas de aprendizaje y el momento de parar)

    punto_control.py   -> guardar un punto de control sin dejar nunca el fichero a medias.
    puntuacion.py      -> la probabilidad que da la red a cada fila y su pérdida media (sin entrenar).
    parada_temprana.py -> ParadaTemprana: recuerda la mejor medida y dice cuándo parar.
    entrenar_red.py    -> ENTRENAMIENTO (ajustes por defecto) y entrenar_red (el bucle de épocas).
    variantes.py       -> EntrenadorVariantes: entrenar una variante con todas sus semillas y rondas, o leerla del
                          registro (F03.05-F03.08 la usaban copiada en cada notebook).
"""



