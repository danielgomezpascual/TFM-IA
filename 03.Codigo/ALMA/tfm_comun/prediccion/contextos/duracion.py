"""Duración del contexto: cuántos segundos de contexto se prueban en cada conjunto, y cuántas ventanas de 4 s son.

Motivo para cambiar: que cambien los contextos candidatos o el tope por conjunto.

Contexto máximo por conjunto
============================
El contexto no pasa del preictal del conjunto (umbral de lead-in - SPH): así TODA crisis que supera el umbral
da al menos una ventana ⏳ con su contexto completo (el tramo de una crisis mide como poco ese preictal).
Modelo general (3 min de umbral: 120 s de preictal) -> hasta 60 s; umbral de 2 min (60 s) -> hasta 60 s;
ausencias (50 s) -> hasta 32 s.

Antes (hasta el 2026-11-06) el tope era un TERCIO del preictal: con el umbral de 2 min dejaba el contexto en
16 s, y F03.03 había visto que mirar más atrás es lo que más ayuda. Se cambió para trabajar como Tang et al.
(2022): la muestra es un clip de 60 s y no tiene por qué caber entero en la zona ⏳.
"""

from tfm_comun.prediccion.umbrales import SPH_POR_DEFECTO_S, UMBRAL_LEADIN_POR_DEFECTO_S, sph_s, umbral_leadin_s


CONTEXTOS_CANDIDATOS_S = [4, 8, 16, 32, 60]   # segundos de contexto que se pueden probar (múltiplos de 4)


FRACCION_MAXIMA = 1.0                         # el contexto no pasa de esta fracción del preictal (antes 1/3)


def preictal_del_conjunto_s(conjunto):
    """Segundos de preictal (umbral de lead-in - SPH) del modelo de un conjunto de F02.04.

    Ejemplos: 'general' -> 120 (umbral de 3 min) · 'umbral_10min' -> 540 · 'grupo_ausencia' -> 50.
    """
    if conjunto.startswith('umbral_') and conjunto.endswith('min'):
        return int(conjunto[len('umbral_'):-len('min')]) * 60 - SPH_POR_DEFECTO_S
    if conjunto == 'grupo_ausencia':
        return umbral_leadin_s('absz') - sph_s('absz')
    return UMBRAL_LEADIN_POR_DEFECTO_S - SPH_POR_DEFECTO_S


def contextos_del_conjunto(conjunto, candidatos=CONTEXTOS_CANDIDATOS_S, fraccion=FRACCION_MAXIMA):
    """Los contextos que se prueban en un conjunto: los candidatos que no pasan de `fraccion` de su preictal.

    Ejemplos: 'general' (120 s) -> [4, 8, 16, 32, 60] · 'umbral_2min' (60 s) -> [4, 8, 16, 32, 60] ·
    'grupo_ausencia' (50 s) -> [4, 8, 16, 32].
    """
    tope = fraccion * preictal_del_conjunto_s(conjunto)
    return [c for c in candidatos if c <= tope + 1e-9]


def ventanas_por_contexto(contexto_s, ventana_s=4):
    """Cuántas ventanas de 4 s forman un contexto: 16 s -> 4 · 60 s -> 15."""
    assert contexto_s % ventana_s == 0, f'El contexto ({contexto_s} s) tiene que ser múltiplo de la ventana ({ventana_s} s).'
    return contexto_s // ventana_s
