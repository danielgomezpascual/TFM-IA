"""
Umbrales de predicción: lead-in, SPH y margen postictal de cada tipo de crisis, y umbrales que se comparan.

Motivo para cambiar: que cambie algún umbral. Si se cambia un umbral, se cambia solo aquí.
Incluye también la duración mínima de una crisis para usarla.

Todo en SEGUNDOS, salvo las listas de umbrales que se comparan, que van en minutos.

- Margen postictal: segundos después de una crisis que no se usan, ni como preictal ni como
  interictal, porque el EEG puede seguir alterado. Depende del tipo de la crisis que ACABA.
- Umbral de lead-in: lead-in útil mínimo (tiempo sin crisis antes, ya descontado el margen
  postictal de la crisis anterior) para usar una crisis en predicción.
- SPH (seizure prediction horizon): segundos justo antes de la crisis que no se usan. Es la
  antelación mínima del aviso. La ventana preictal máxima es umbral de lead-in - SPH.

El modelo general y los tipos sin valores propios usan los valores por defecto.

Imagen: tfm_comun/imagenes/umbrales_periodos.png (los periodos con los valores por defecto y con los de absz)
"""

# Valores por defecto (modelo general y tipos sin valores propios)
MARGEN_POSTICTAL_POR_DEFECTO_S = 300   # 5 min
UMBRAL_LEADIN_POR_DEFECTO_S = 240      # 4 min: 3 de preictal + 1 de SPH (elegido en fase_02/01_seleccion_crisis, sección 4.3)
SPH_POR_DEFECTO_S = 60                 # 1 min

# Tipos con valores propios
UMBRALES_TIPO = {
    # Las ausencias son cortas y muy seguidas: con 5 min casi no quedan crisis, así que se afina en segundos
    'absz': {
        'margen_postictal_s': 0,   # la ausencia casi no deja postictal: el paciente vuelve de golpe
        'umbral_leadin_s': 60,     # recupera los 12 pacientes de train + dev
        'sph_s': 10,               # inicio brusco: basta un margen corto; deja 50 s de preictal
    },
}

# Duración mínima de una crisis (s). Las más cortas se descartan: no llenan ni una ventana de 2 s,
# la más pequeña que se maneja (fase_01/03v, duración por tipo), así que no sirven para clasificar su tipo.
DURACION_MINIMA_CRISIS_S = 2

# Umbrales de lead-in que se comparan en la Fase 3 y se recuentan en el EDA (minutos). El más alto fija también la distancia
# mínima del interictal antes de una crisis, para que ningún interictal sea preictal con ningún umbral.
UMBRALES_LEADIN_COMPARACION_MIN = [2, 3, 4, 5, 7, 10, 15]


def margen_postictal_s(tipo):
    """Margen postictal (s) de un tipo de crisis: segundos DESPUÉS de la crisis que no se usan.

    Tras una crisis el EEG puede seguir alterado (lentitud, aplanamiento), así que ese trozo no vale
    ni como preictal de la siguiente crisis ni como interictal. Busca el valor propio del tipo en
    UMBRALES_TIPO; si no lo tiene, usa MARGEN_POSTICTAL_POR_DEFECTO_S (300 s).
    Ejemplos: 'fnsz' -> 300 · 'absz' -> 0 (la ausencia casi no deja postictal).

    Imagen: tfm_comun/imagenes/umbrales_periodos.png
    """
    return UMBRALES_TIPO.get(tipo, {}).get('margen_postictal_s', MARGEN_POSTICTAL_POR_DEFECTO_S)


def umbral_leadin_s(tipo):
    """Umbral de lead-in (s) de un tipo: lead-in útil mínimo para usar una crisis en predicción.

    Lead-in útil = tiempo sin crisis antes de esta, ya descontado el margen postictal de la anterior.
    Si no llega al umbral, no hay sitio para su preictal + SPH y la crisis no se usa para predecir
    (sí para clasificar su tipo). umbral = preictal máximo + SPH.
    Ejemplos: 'fnsz' -> 240 (3 min de preictal + 1 de SPH) · 'absz' -> 60 (50 s + 10 s).

    Imagen: tfm_comun/imagenes/umbrales_periodos.png
    """
    return UMBRALES_TIPO.get(tipo, {}).get('umbral_leadin_s', UMBRAL_LEADIN_POR_DEFECTO_S)


def sph_s(tipo):
    """SPH (seizure prediction horizon, en s) de un tipo: segundos justo ANTES de la crisis que no se usan.

    Es la antelación mínima del aviso (el tiempo para reaccionar) y evita que el modelo "prediga"
    con actividad que ya es el comienzo de la crisis.
    Ejemplos: 'fnsz' -> 60 · 'absz' -> 10 (inicio brusco: basta un margen corto).

    Imagen: tfm_comun/imagenes/umbrales_periodos.png
    """
    return UMBRALES_TIPO.get(tipo, {}).get('sph_s', SPH_POR_DEFECTO_S)
