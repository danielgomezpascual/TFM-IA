"""
Nombres de fichero: convertir un texto cualquiera (con iconos, espacios o tildes) en un nombre de fichero seguro.

Motivo para cambiar: que cambie qué caracteres se permiten en los nombres de los ficheros que se guardan.
"""

import re


def nombre_fichero(texto):
    """Un texto convertido en nombre de fichero: todo lo que no sea letra, número o '_' pasa a '_'.

    Ejemplos: '🟡 dev' -> 'dev' · 'pliegue 2' -> 'pliegue_2' · 'semilla 42 · dev' -> 'semilla_42_dev'.
    """
    return re.sub(r'[^\w]+', '_', texto).strip('_')
