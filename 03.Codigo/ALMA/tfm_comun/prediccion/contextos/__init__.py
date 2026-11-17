"""Contexto: lo que pasó en los segundos ANTERIORES a una ventana, resumido para que lo use un modelo.


Imagen: fase_03/img/contexto_ventanas.png (la ventana actual, las anteriores y su resumen)

Qué es el contexto
==================
La ventana sigue siendo de 4 s (las ausencias y la Fase 4 la necesitan así). El contexto son los C segundos
que ACABAN donde acaba la ventana, cortados en trozos de 4 s seguidos y sin solape:

    contexto de 16 s = la ventana actual + las 3 de 4 s justo antes (4 ventanas)
    contexto de 60 s = la ventana actual + las 14 de antes (15 ventanas)

Se sacan siempre de la señal de F02.02 (el tramo de la ventana, que está completo), y no de los paquetes:
F02.04 sorteó las 💤 interictales para equilibrar, así que en los paquetes a muchas les faltan las vecinas.
Así el contexto se construye IGUAL para ⏳ y para 💤 (si no, sería una pista tramposa).

Una ventana solo tiene contexto completo si (1) el tramo empieza al menos C - 4 s antes que ella y (2) el
contexto no entra en los primeros DESCARTE_INICIO_FICHERO_S segundos del fichero (el arranque del filtro, la
misma regla que F02.03 aplica a las ventanas). La regla es la misma para las dos clases. Para comparar
contextos de forma justa, todos se miden con las MISMAS ventanas: las que tienen el contexto más largo del estudio.

El contexto PUEDE empezar antes de la zona ⏳ del modelo (como el clip de Tang et al., 2022): el tramo ⏳ de
F02.01 va desde el final del margen postictal (o el inicio del fichero) hasta el SPH, así que la señal anterior
a la zona ⏳ existe y está limpia. La etiqueta la da SOLO la ventana actual (el final del clip).

Cómo se resume (de K ventanas x 110 números a 330 números)
==========================================================
Para cada una de las 110 características (caracteristicas.py):
    - media: el valor típico en los últimos C segundos;
    - variación: su desviación típica (¿se ha movido mucho?);
    - cambio: la ventana actual menos la media (¿lo de ahora es distinto de lo de antes?).
Con contexto de 4 s (K = 1) la variación y el cambio son 0: es exactamente la línea base de F03.02.

Contexto máximo por conjunto
============================
El contexto no pasa del preictal del conjunto (umbral de lead-in - SPH): así TODA crisis que supera el umbral
da al menos una ventana ⏳ con su contexto completo (el tramo de una crisis mide como poco ese preictal).
Modelo general (3 min de umbral: 120 s de preictal) -> hasta 60 s; umbral de 2 min (60 s) -> hasta 60 s;
ausencias (50 s) -> hasta 32 s.

Antes (hasta el 2026-11-06) el tope era un TERCIO del preictal: con el umbral de 2 min dejaba el contexto en
16 s, y F03.03 había visto que mirar más atrás es lo que más ayuda. Se cambió para trabajar como Tang et al.
(2022): la muestra es un clip de 60 s y no tiene por qué caber entero en la zona ⏳.

Cada pieza en su fichero (un solo motivo para cambiar cada una):
    duracion.py      -> cuántos segundos de contexto se prueban en cada conjunto (y cuántas ventanas son).
    disponibilidad.py -> qué ventanas tienen su contexto completo y limpio, y qué ventanas hacen falta calcular.
    calculo.py       -> lo que se mide de cada ventana del contexto, leyéndola de la señal de F02.02.
    indices.py       -> dónde está, en el almacén, cada ventana del contexto de cada fila (sin copiar datos).
    resumen.py       -> de K ventanas x 110 números a 330: media, variación y cambio.
"""



