"""Lectura de una diferencia: cómo se dice en palabras "A - B = 0,03 [0,01 – 0,05]".

Motivo para cambiar: que cambien los textos o la mejora mínima que se considera útil.

Las tres lecturas posibles
==========================
    el intervalo está por ENCIMA de 0  -> 'mejora real' (y, si es menor que la mejora mínima, "pero pequeña");
    el intervalo está por DEBAJO de 0  -> 'empeora (real)';
    el intervalo INCLUYE el 0          -> 'sin diferencia clara': con estos pacientes, puede ser casualidad.
"""

from tfm_comun.visual.numeros import decimal


MEJORA_MINIMA = 0.02   # por debajo de 2 centésimas de AUC, una mejora "real" se considera pequeña


def lectura_diferencia(diferencia, mejora_minima=MEJORA_MINIMA):
    """La lectura en palabras de (diferencia, inferior, superior), de diferencia_auroc_... o de Evaluador.diferencia.

    Ejemplos (mejora mínima 0,02):
        (0,04, 0,01, 0,07)   -> 'mejora real'
        (0,01, 0,002, 0,02)  -> 'mejora real, pero pequeña (< 0,02)'
        (0,01, -0,02, 0,04)  -> 'sin diferencia clara (el intervalo incluye el 0)'
        (-0,03, -0,05, -0,01) -> 'empeora (real)'
    """
    valor, inferior, superior = diferencia
    if inferior > 0:
        return 'mejora real' + ('' if valor >= mejora_minima else f', pero pequeña (< {decimal(mejora_minima)})')
    return 'empeora (real)' if superior < 0 else 'sin diferencia clara (el intervalo incluye el 0)'
