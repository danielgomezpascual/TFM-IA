"""
Estudio: qué conjunto de F02.04 se estudia en TODA la Fase 3.

Motivo para cambiar: que se quiera estudiar otro conjunto (otro umbral de lead-in, un grupo de inicio...).

Antes cada notebook de la Fase 3 llevaba su propio CONJUNTO y había que cambiarlo en todos. Ahora se cambia
SOLO aquí. La celda de parámetros de cada notebook deja CONJUNTO = None, que significa "el de este fichero";
papermill (o una prueba a mano) puede seguir poniendo otro valor en un notebook concreto.

Conjuntos (los crea F02.04):
    'general'                       modelo general con el umbral de lead-in por defecto (umbrales.py: 3 min)
    'umbral_2min', 'umbral_4min'... el mismo modelo con otro umbral, para comparar (UMBRALES_LEADIN_COMPARACION_MIN)
    'grupo_focal', 'grupo_generalizada', 'grupo_ausencia'   un modelo por grupo de inicio

Los resultados de cada conjunto se guardan aparte (outputs/fase_03/<conjunto>/<notebook>/), así que estudiar
otro conjunto no borra lo del anterior. Los paquetes (F03.01) y las características (F03.02) ya iban por conjunto.
"""

CONJUNTO_ESTUDIO = 'general'


def conjunto_del_estudio(conjunto=None):
    """El conjunto que usa un notebook: el que se le pase o, si es None, CONJUNTO_ESTUDIO.

    Ejemplos: conjunto_del_estudio() -> 'general' · conjunto_del_estudio('umbral_2min') -> 'umbral_2min'.
    """
    return conjunto or CONJUNTO_ESTUDIO
