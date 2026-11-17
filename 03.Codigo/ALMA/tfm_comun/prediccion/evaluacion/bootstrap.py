"""Bootstrap por paciente: los ajustes comunes de todos los intervalos de confianza de la evaluación.

Motivo para cambiar: que cambie el número de repeticiones o el nivel de confianza.

Por qué un intervalo POR PACIENTE
=================================
Las ventanas de un mismo paciente se parecen mucho entre sí: 10.000 ventanas de 30 pacientes no son 10.000 datos
independientes, son más bien 30. Si el intervalo se calcula sorteando ventanas, sale estrechísimo y engaña. Se
sortean PACIENTES (bootstrap por paciente): cada repetición elige pacientes al azar, con repetición, y recalcula la
nota con todas sus ventanas. El intervalo dice cuánto cambiaría la cifra con otros pacientes parecidos.

Imagen: fase_02/img/bootstrap_por_paciente.png
"""


N_BOOTSTRAP = 500    # repeticiones del bootstrap: 500 basta para un intervalo del 95 % estable a 2 decimales


NIVEL = 0.95         # intervalo de confianza del 95 %
