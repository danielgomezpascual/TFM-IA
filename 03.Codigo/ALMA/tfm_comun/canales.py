"""
Catálogo de canales de TUSZ: qué es cada señal del EDF.

Motivo para cambiar: que aparezcan canales nuevos o cambie cómo se clasifican.
Es el equivalente, para los canales, de lo que son las etiquetas de crisis:
aquí se decide el TIPO de cada canal y la REGIÓN de cada electrodo; sus
iconos están en iconos.py y sus colores en colores.py.

Nombres en el EDF: 'EEG FP1-LE', 'EEG T3-REF', 'PHOTIC PH', 'DC1-DC'...
    - prefijo 'EEG ': lo pone el equipo a casi todo, aunque no sea EEG (p. ej. 'EEG EKG-LE').
    - sufijo: la referencia con la que se midió (ver REFERENCIAS).
"""

import re

# Referencias: respecto a qué se mide cada electrodo (sufijo del nombre en el EDF)
REFERENCIAS = {
    'LE': 'orejas unidas (linked ears): media de A1 y A2',
    'REF': 'referencia promedio (AR, averaged reference): media de todos los electrodos',
}

# Montajes de TUSZ (nombre de la carpeta del EDF)
MONTAJES_TUSZ = {
    '01_tcp_ar': 'referencia promedio (sufijo -REF), TCP de 22 canales',
    '02_tcp_le': 'orejas unidas (sufijo -LE), TCP de 22 canales',
    '03_tcp_ar_a': 'referencia promedio sin A1/A2, TCP de 20 canales',
    '04_tcp_le_a': 'orejas unidas sin A1/A2 (documentado, no aparece en v2.0.6)',
}

# Electrodos del sistema 10-20 que aparecen en TUSZ y región de la cabeza de cada uno
REGION_ELECTRODO = {
    'FP1': 'frontopolar', 'FP2': 'frontopolar',
    'F7': 'frontal', 'F3': 'frontal', 'FZ': 'frontal', 'F4': 'frontal', 'F8': 'frontal',
    'C3': 'central', 'CZ': 'central', 'C4': 'central',
    'T1': 'temporal', 'T2': 'temporal', 'T3': 'temporal', 'T4': 'temporal',
    'T5': 'temporal', 'T6': 'temporal',
    'P3': 'parietal', 'PZ': 'parietal', 'P4': 'parietal',
    'O1': 'occipital', 'OZ': 'occipital', 'O2': 'occipital',
    'A1': 'auricular', 'A2': 'auricular',
}
ELECTRODOS_1020 = set(REGION_ELECTRODO)
REGIONES = ['frontopolar', 'frontal', 'central', 'temporal', 'parietal', 'occipital', 'auricular']

# Tipos de canal (los de MNE) y qué son en TUSZ
DESCRIPCION_TIPO = {
    'eeg': 'electrodo de cuero cabelludo del sistema 10-20',
    'ecg': 'electrocardiograma (actividad del corazón)',
    'eog': 'electrooculograma (movimientos de los ojos)',
    'emg': 'electromiograma (actividad muscular)',
    'stim': 'marcador de estímulo (fotoestimulación con luz)',
    'resp': 'respiración',
    'misc': 'otros: electrodos especiales, canales sin uso o calculados por el equipo',
}

# Canales 'misc' con nombre conocido
DESCRIPCION_MISC = {
    'PG1': 'electrodo nasofaríngeo', 'PG2': 'electrodo nasofaríngeo',
    'SP1': 'electrodo esfenoidal', 'SP2': 'electrodo esfenoidal',
    'C3P': 'electrodo extra (fuera del 10-20)', 'C4P': 'electrodo extra (fuera del 10-20)',
    'IBI': 'calculado por el equipo (intervalo entre ráfagas)',
    'BURSTS': 'calculado por el equipo (ráfagas)',
    'SUPPR': 'calculado por el equipo (supresión)',
}


def referencia_canal(nombre_original):
    """'EEG FP1-LE' -> 'LE', 'EEG T3-REF' -> 'REF'; None si no lleva referencia."""
    encontrado = re.search(r'-(LE|REF)$', nombre_original.strip())
    return encontrado.group(1) if encontrado else None


def limpiar_nombre_canal(nombre):
    """'EEG FP1-LE' -> 'FP1', 'EEG T3-REF' -> 'T3', 'PHOTIC PH' -> 'PHOTIC', 'DC1-DC' -> 'DC1'."""
    nombre = re.sub(r'^EEG\s+', '', nombre.strip())
    nombre = re.sub(r'(-LE|-REF|-DC|\s+PH)$', '', nombre)
    return nombre.upper()


def tipo_canal(nombre):
    """Tipo de canal de MNE a partir del nombre limpio."""
    if nombre in ELECTRODOS_1020:
        return 'eeg'
    if nombre.startswith(('EKG', 'ECG')):
        return 'ecg'
    if nombre in ('LOC', 'ROC') or nombre.startswith('EOG'):
        return 'eog'
    if nombre.startswith('EMG'):
        return 'emg'
    if nombre.startswith('PHOTIC'):
        return 'stim'
    if nombre.startswith('RESP'):
        return 'resp'
    return 'misc'


def region_electrodo(nombre):
    """Región de la cabeza de un electrodo 10-20 ('FP1' -> 'frontopolar'); None si no lo es."""
    return REGION_ELECTRODO.get(nombre)


def descripcion_canal(nombre):
    """Texto corto que explica qué es un canal (nombre limpio)."""
    tipo = tipo_canal(nombre)
    if tipo != 'misc':
        return DESCRIPCION_TIPO[tipo]
    if nombre in DESCRIPCION_MISC:
        return DESCRIPCION_MISC[nombre]
    if nombre.isdigit():
        return 'entrada del amplificador sin electrodo (sin uso)'
    if nombre.startswith('DC'):
        return 'entrada de corriente continua del equipo (sin uso)'
    return 'desconocido'


def lado_electrodo(nombre):
    """Lado de la cabeza de un electrodo 10-20: número impar -> 'izquierdo' (FP1, C3),
    par -> 'derecho' (FP2, C4), 'Z' -> 'línea media' (FZ, CZ). A1 es izquierdo y A2 derecho."""
    if nombre.endswith('Z'):
        return 'línea media'
    return 'izquierdo' if int(nombre[-1]) % 2 else 'derecho'
