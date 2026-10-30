"""
Versiones de Python y de las librerías que usa cada notebook.

Motivo para cambiar: que cambie cómo se muestran o se guardan las versiones.
Se muestran al principio de cada notebook (después de los imports) para saber con qué
versiones se generaron los resultados, y se guardan en su configuracion.json cuando lo hay.
"""

import platform


def versiones(*librerias):
    """Versión de Python y de cada librería que se le pasa, en un diccionario.

    Lee platform.python_version() y el atributo __version__ de cada módulo.
    Ejemplo: versiones(np, pd) -> {'Python': '3.14.0', 'numpy': '2.5.4', 'pandas': '3.0.6'}
    """
    return {'Python': platform.python_version(), **{lib.__name__: lib.__version__ for lib in librerias}}


def mostrar_versiones(*librerias):
    """Imprime las versiones (una por línea, alineadas) y las devuelve para guardarlas en configuracion.json.

    Es la función común que se llama al principio de cada notebook, justo después de los imports,
    para saber con qué versiones se generaron los resultados.
    Uso: versiones_usadas = mostrar_versiones(np, pd, matplotlib)
    """
    resultado = versiones(*librerias)
    for libreria, version in resultado.items():
        print(f'{libreria:<11} {version}')
    return resultado
