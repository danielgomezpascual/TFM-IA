"""
Iconos para textos y tablas: etiquetas, particiones, tipos de canal y regiones.

Motivo para cambiar: que cambie la convención de iconos del TFM.
Si se cambia un icono, se cambia solo aquí.
"""

# Iconos de las etiquetas de TUSZ
ICONO_ETIQUETA = {
    'bckg': '⬜',      # fondo, sin crisis
    'seiz': '⬛',      # crisis sin tipo (.csv_bi)
    'sin_tipo': '⬛',  # crisis del .csv_bi a la que no se pudo asignar tipo
    'fnsz': '🟥',      # focal no específica
    'gnsz': '🟦',      # generalizada no específica
    'cpsz': '🟧',      # parcial compleja
    'absz': '🟩',      # ausencia
    'tcsz': '🟪',      # tónico-clónica
    'tnsz': '🟨',      # tónica
    'spsz': '🟫',      # parcial simple
    'mysz': '🔳',      # mioclónica
}

# Iconos de las particiones ('todas' = dataset completo; 🔴 se reserva para avisos)
ICONO_SPLIT = {'train': '🔵', 'dev': '🟡', 'eval': '🟢', 'todas': '⚪'}

# Iconos de los tipos de canal (tipos de MNE, ver canales.py)
ICONO_TIPO_CANAL = {
    'eeg': '🧠',   # electrodo de cuero cabelludo 10-20
    'ecg': '🫀',   # electrocardiograma
    'eog': '👁️',   # movimientos oculares
    'emg': '💪',   # actividad muscular
    'stim': '💡',  # fotoestimulación
    'resp': '🫁',  # respiración
    'misc': '❔',  # otros
}

# Iconos de las regiones de la cabeza: cuadrados del mismo color que en las figuras
ICONO_REGION = {
    'frontopolar': '🟪', 'frontal': '🟦', 'central': '🟩', 'temporal': '🟧',
    'parietal': '🟨', 'occipital': '🟥', 'auricular': '🟫',
}

_ICONO_TEXTO = {**ICONO_ETIQUETA, **ICONO_TIPO_CANAL, **ICONO_REGION}


def con_icono(texto):
    """Añade el icono a una etiqueta ('absz' -> '🟩 absz'), a un tipo de canal
    ('eeg' -> '🧠 eeg'), a una región ('temporal' -> '🟧 temporal') o a un texto
    que empiece por una partición ('dev crisis' -> '🟡 dev crisis')."""
    texto = str(texto)
    if texto in _ICONO_TEXTO:
        return f'{_ICONO_TEXTO[texto]} {texto}'
    for split, icono in ICONO_SPLIT.items():
        if texto.startswith(split):
            return f'{icono} {texto}'
    return texto


def tabla_con_iconos(df, columnas=()):
    """Copia de una tabla de pandas con iconos en filas, columnas y en los valores
    de las columnas indicadas. Solo para mostrar: los CSV se guardan sin iconos."""
    df = df.rename(index=con_icono, columns=con_icono)
    for columna in columnas:
        df[con_icono(columna)] = df[con_icono(columna)].map(con_icono)
    return df
