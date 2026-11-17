"""Huella de una configuración: 10 caracteres que la identifican (si cambia cualquier número, cambia la huella).

Motivo para cambiar: que cambie cómo se identifica una configuración.

Cómo funciona: el diccionario se convierte en texto JSON con las claves ordenadas (así el orden en que se escriben
no importa) y se le calcula el SHA-1, una "huella digital" de un texto: dos textos distintos dan, en la práctica,
huellas distintas. Se guardan los 10 primeros caracteres: más que suficiente para unos cientos de experimentos.
"""

import hashlib
import json


def huella(configuracion):
    """10 caracteres que identifican una configuración (el mismo diccionario -> la misma huella, en cualquier orden)."""
    texto = json.dumps(configuracion, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(texto.encode('utf-8')).hexdigest()[:10]
