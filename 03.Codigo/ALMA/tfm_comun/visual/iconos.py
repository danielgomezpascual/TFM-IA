"""
Iconos para textos, tablas y leyendas: etiquetas, grupos de inicio, particiones, clases, tipos de canal y regiones.

Motivo para cambiar: que cambie la convención de iconos del TFM.
Si se cambia un icono, se cambia solo aquí.

Cada tipo de crisis lleva detrás, entre paréntesis, el icono de su grupo de inicio: 🟥 fnsz (♦️).
"""

from tfm_comun.dataset.tipos_crisis import grupo

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

# Iconos de los grupos de inicio (ver tipos_crisis.py): rombos, para no confundirlos con los
# cuadrados de las etiquetas ni con los círculos de las particiones
ICONO_GRUPO = {
    'focal': '♦️',         # rojo, como 🟥 fnsz, el tipo focal principal
    'generalizada': '🔷',  # azul, como 🟦 gnsz
    'ausencia': '🔹',      # azul pequeño: generalizada, pero va aparte
}

# Iconos de las particiones ('todas' = dataset completo; 🔴 se reserva para avisos)
ICONO_SPLIT = {'train': '🔵', 'dev': '🟡', 'eval': '🟢', 'todas': '⚪'}

# Iconos de las clases de los tramos y ventanas (fase_02): triángulos planos que se van rellenando
# según se acerca la crisis. Son caracteres de texto (no emojis), así que también se ven bien en matplotlib.
ICONO_CLASE = {
    'interictal': '△',   # vacío: lejos de cualquier crisis
    'preictal': '◭',     # medio lleno: se acerca una crisis
    'ictal': '▲',        # lleno: la crisis
}

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

_ICONO_TEXTO = {**ICONO_GRUPO, **ICONO_CLASE, **ICONO_TIPO_CANAL, **ICONO_REGION}


def icono_etiqueta(etiqueta):
    """Texto de una etiqueta de TUSZ con su icono delante y el de su grupo de inicio detrás.

    El cuadrado de color dice qué tipo es; el rombo entre paréntesis, a qué grupo pertenece
    (grupo() de tipos_crisis.py). Las etiquetas sin grupo (bckg, seiz, sin_tipo) van sin paréntesis.
    Ejemplos: 'fnsz' -> '🟥 fnsz (♦️)' · 'absz' -> '🟩 absz (🔹)' · 'bckg' -> '⬜ bckg'.
    """
    nombre_grupo = grupo(etiqueta)
    sufijo = f' ({ICONO_GRUPO[nombre_grupo]})' if nombre_grupo else ''
    return f'{ICONO_ETIQUETA[etiqueta]} {etiqueta}{sufijo}'


def con_icono(texto):
    """Añade el icono que le toca a cualquier texto del TFM; si no reconoce el texto, lo devuelve igual.

    Mira, por este orden:
        1. ¿Es una etiqueta de crisis?   'absz'       -> '🟩 absz (🔹)'
        2. ¿Es un grupo, clase, tipo de canal o región?  'focal' -> '♦️ focal' · 'preictal' -> '◭ preictal' ·
           'eeg' -> '🧠 eeg' · 'temporal' -> '🟧 temporal'
        3. ¿Empieza por una partición?   'dev crisis' -> '🟡 dev crisis'
    Acepta cualquier valor (lo pasa a texto), así que se puede usar con df.rename(index=con_icono).
    """
    texto = str(texto)
    if texto in ICONO_ETIQUETA:
        return icono_etiqueta(texto)
    if texto in _ICONO_TEXTO:
        return f'{_ICONO_TEXTO[texto]} {texto}'
    for split, icono in ICONO_SPLIT.items():
        if texto.startswith(split):
            return f'{icono} {texto}'
    return texto


def tabla_con_iconos(df, columnas=()):
    """Copia de una tabla de pandas con iconos, solo para mostrarla (los CSV se guardan sin iconos).

    Parámetros:
        df: la tabla original (no se modifica).
        columnas: columnas cuyos VALORES también llevan icono (p. ej. ['split', 'tipo']).
            Las etiquetas de filas y columnas llevan icono siempre (también las de varios niveles).

    Devuelve:
        Una copia de df con los iconos puestos con con_icono().
    """
    df = df.rename(index=con_icono, columns=con_icono)
    for columna in columnas:
        df[con_icono(columna)] = df[con_icono(columna)].map(con_icono)
    return df
