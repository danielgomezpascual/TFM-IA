"""
Utilidades comunes a todos los notebooks del TFM.

Contiene la convención visual (iconos por etiqueta y por partición, y colores
de las figuras) para que sea la misma en todo el proyecto. Si se cambia un
icono o un color, se cambia solo aquí.
"""

from matplotlib import ticker

SPLITS = ['train', 'dev', 'eval']

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

# Colores de las figuras, a juego con los iconos de partición
# (matplotlib no dibuja bien los emojis, así que en las figuras se usan colores)
COLORES_SPLIT = {'train': 'tab:blue', 'dev': 'goldenrod', 'eval': 'tab:green'}


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


def _numero_legible(valor, _posicion=None):
    """Número en formato español: 1000 -> '1.000', 0.5 -> '0,5'."""
    if valor >= 1:
        return f'{valor:,.0f}'.replace(',', '.')
    return f'{valor:g}'.replace('.', ',')


def eje_log_legible(ax, eje='y'):
    """En un eje con escala logarítmica, muestra números normales
    (1, 2, 5, 10, 20, 50, 100, 1.000...) en vez de potencias de 10 (10², 10³)."""
    eje_mpl = ax.yaxis if eje == 'y' else ax.xaxis
    eje_mpl.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    eje_mpl.set_major_formatter(ticker.FuncFormatter(_numero_legible))
    eje_mpl.set_minor_formatter(ticker.NullFormatter())
