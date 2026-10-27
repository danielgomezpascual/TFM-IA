"""
Versiones de Python y de las librerías que usa cada notebook.

Motivo para cambiar: que cambie cómo se muestran o se guardan las versiones.
Se muestran al principio de cada notebook (después de los imports) para saber con qué
versiones se generaron los resultados, y se guardan en su configuracion.json cuando lo hay.
"""

import platform


def versiones(*librerias):
    """Diccionario {'Python': '3.12.1', 'numpy': '2.1.0', ...} con los módulos que se le pasan."""
    return {'Python': platform.python_version(), **{lib.__name__: lib.__version__ for lib in librerias}}


def mostrar_versiones(*librerias):
    """Muestra las versiones (una por línea) y las devuelve para guardarlas en la configuración.
    Uso: versiones_usadas = mostrar_versiones(np, pd, matplotlib)"""
    resultado = versiones(*librerias)
    for libreria, version in resultado.items():
        print(f'{libreria:<11} {version}')
    return resultado
