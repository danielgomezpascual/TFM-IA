"""
Configuración de otro notebook: leer el configuracion.json que dejó un notebook anterior y comprobar que vale.

Motivo para cambiar: que cambie cómo se lee o se comprueba la configuración de un notebook anterior.

Por qué hace falta
==================
Los notebooks son dinámicos: cada uno lee lo que dejaron los anteriores (semilla, contextos, decisiones, notas) en
su configuracion.json, en vez de llevar cifras escritas a mano. Hay dos errores típicos que conviene parar en seco,
con un mensaje que diga qué hacer:
    1. el notebook anterior no se ha ejecutado (no está el fichero);
    2. se ejecutó con OTRO conjunto (p. ej. 'umbral_2min' en vez de 'general'): sus cifras no valen para este.
Antes cada notebook de la Fase 3 llevaba su copia de esta función (leer_config); ahora está aquí una sola vez.
"""

import json


def leer_configuracion(ruta, notebook, conjunto, obligatorio=True, pista=''):
    """El configuracion.json de un notebook anterior, comprobando que es del mismo conjunto.

    Parámetros:
        ruta: el configuracion.json.  notebook: cómo se llama, para el mensaje (p. ej. 'F03.01 (fase_03/01_datos_red.ipynb)').
        conjunto: el conjunto de este notebook ('general', 'umbral_2min'...).
        obligatorio: True = si no existe, para con un error; False = si no existe, devuelve None (p. ej. un notebook
            con el que solo se compara si está ejecutado).
        pista: texto que se añade al mensaje si falta (p. ej. ' y MODO_PRUEBA = True').

    Devuelve:
        El diccionario de la configuración, o None (solo si obligatorio=False y no existe).

    Ejemplo:
        config_01 = leer_configuracion(RUTA_OUTPUTS_FASE_03 / CONJUNTO / '01_datos_red' / 'configuracion.json',
                                       'F03.01 (fase_03/01_datos_red.ipynb)', CONJUNTO)
    """
    if not ruta.exists():
        assert not obligatorio, f'No encuentro {ruta}. Ejecuta antes {notebook} con CONJUNTO = {conjunto!r}{pista}.'
        return None
    config = json.loads(ruta.read_text(encoding='utf-8'))
    assert config['conjunto'] == conjunto, f'{notebook} se ejecutó con CONJUNTO = {config["conjunto"]!r}: ejecútalo con {conjunto!r}.'
    return config
