"""
Iconos de etiquetas y particiones para textos y tablas.

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


def con_icono(texto):
    """Añade el icono a una etiqueta ('absz' -> '🟩 absz') o a un texto que
    empiece por una partición ('dev crisis' -> '🟡 dev crisis')."""
    texto = str(texto)
    if texto in ICONO_ETIQUETA:
        return f'{ICONO_ETIQUETA[texto]} {texto}'
    for split, icono in ICONO_SPLIT.items():
        if texto.startswith(split):
            return f'{icono} {texto}'
    return texto


def tabla_con_iconos(df):
    """Copia de una tabla de pandas con iconos en filas y columnas.
    Solo para mostrar: los CSV se guardan sin iconos."""
    return df.rename(index=con_icono, columns=con_icono)
