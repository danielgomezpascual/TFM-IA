"""Variables de atajo: qué pistas se vigilan y cómo se calcula cada una para cada ventana.

Motivo para cambiar: que cambien las pistas que se vigilan o cómo se calculan.

Todo trabaja con tablas (sin leer la señal): es rápido y no depende del modelo que se entrene después.
"""

import numpy as np

from tfm_comun.senal.remuestreo import FS_OBJETIVO


# ── Las variables que se vigilan ───────────────────────────────────────────────────────────────────
# nombre -> (qué es, cómo podría colarse en el modelo, tipo, ¿la red la ve directamente en la ventana?)
VARIABLES_ATAJO = {
    'canales_a_cero': ('Canales TCP a 0 porque al EDF le faltan A1/A2 (montaje 03_tcp_ar_a)',
                       'La red VE dos filas planas en la ventana: si salen más en una clase, es una pista gratis',
                       'numerica', True),
    'montaje': ('Montaje original del EDF (01_tcp_ar, 02_tcp_le, 03_tcp_ar_a)',
                'Cada referencia deja una "huella" algo distinta aunque se pase todo a TCP',
                'categorica', True),
    'fs_original_hz': ('Frecuencia de muestreo original del equipo (250, 256, 400, 512, 1000 Hz)',
                       'Equipo distinto = filtro antialiasing y ruido distintos, aunque se remuestree a 256 Hz',
                       'categorica', True),
    'nivel_senal_uv': ('Desviación típica mediana de los canales del tramo (µV)',
                       'Amplitud "de fondo" del registro; la normalización por ventana borra casi toda',
                       'numerica', True),
    'minuto_en_fichero': ('Minuto del fichero EDF en el que empieza la ventana',
                          'En TUSZ las crisis suelen estar al principio de su fichero, así que el ⏳ preictal también; '
                          'si el EEG de esa parte del fichero es distinto, la red podría notarlo',
                          'numerica', False),
    'duracion_fichero_min': ('Duración total del fichero EDF (min)',
                             'Ficheros largos y cortos se graban en situaciones distintas (rutina frente a UCI)',
                             'numerica', False),
    'fichero_con_crisis': ('1 si la ventana sale de un fichero que tiene alguna crisis, 0 si no',
                           'Un día con crisis puede ser distinto (sueño, medicación, UCI) aunque la ventana esté lejos de la crisis',
                           'numerica', False),
}


# ── Resolución de la posición en el fichero ──────────────────────────────────────────────────────────
# El minuto se redondea a 30 s. Con el valor exacto había una FUGA: las 💤 se cortan desde el principio del
# tramo y empiezan en segundos múltiplos de 4 (0, 4, 8...); las ⏳ se cortan desde la crisis y empiezan en
# segundos "con decimales". Un modelo separaba las clases casi perfecto solo con eso (AUROC > 0,9), una pista
# que la red no ve en el EEG. A 30 s solo queda lo que importa: en qué parte del fichero está la ventana.
RESOLUCION_POSICION_S = 30


def metadatos_de_ventanas(conjunto, senales, edfs, fs=FS_OBJETIVO):
    """Añade a cada ventana de un conjunto sus variables de atajo (VARIABLES_ATAJO).

    Qué hace:
        Une cada ventana con su tramo (senales.csv, por 'ruta') y con su fichero (edfs.csv, por 'file'),
        y calcula:
            canales_a_cero       = ceros de mascara_canales del tramo (canales que no se pudieron construir)
            montaje, fs_original_hz (del tramo)
            nivel_senal_uv       = desviacion_mediana_uv del tramo
            minuto_en_fichero    = (inicio del tramo en el fichero + muestra_inicio / fs) / 60, redondeado hacia
                                   abajo a RESOLUCION_POSICION_S (30 s) para no dejar la fuga de la rejilla de corte
            duracion_fichero_min = duración del EDF / 60
            fichero_con_crisis   = 1 si el tramo NO es de un 'fichero_sin_crisis'
        Comprueba que ninguna ventana se queda sin tramo o sin fichero (si no, para con un error).

    Parámetros:
        conjunto: tabla de 04_particiones_y_balanceo/conjuntos/<modelo>.csv.
        senales: tabla de 02_preprocesado/senales.csv, leída con dtype={'mascara_canales': str} (si no, pandas
            la lee como un número de 22 cifras y pierde los ceros).
        edfs: tabla de fase_01/01_inventario/edfs.csv.
        fs: frecuencia de la señal guardada (256 Hz), para pasar muestras a segundos.

    Devuelve:
        Copia del conjunto con las 7 variables añadidas (una fila por ventana, mismo orden).
    """
    tramos = senales[['ruta', 'montage', 'fs_original', 'mascara_canales', 'inicio_s', 'origen', 'desviacion_mediana_uv']]
    v = conjunto.merge(tramos, on='ruta', how='left', validate='many_to_one')
    v = v.merge(edfs[['file', 'edf_dur_s']], on='file', how='left', validate='many_to_one')
    faltan = v[['montage', 'edf_dur_s']].isna().any(axis=1)
    assert not faltan.any(), f'🔴 {int(faltan.sum())} ventanas sin tramo en senales.csv o sin fichero en edfs.csv.'

    mascara = v.mascara_canales.astype(str)
    assert mascara.str.fullmatch(r'[01]+').all(), \
        "🔴 mascara_canales no es texto de 0 y 1: lee senales.csv con dtype={'mascara_canales': str}."
    v['canales_a_cero'] = mascara.str.count('0')
    v['montaje'] = v.montage
    v['fs_original_hz'] = v.fs_original.round().astype(int)
    v['nivel_senal_uv'] = v.desviacion_mediana_uv
    segundo = v.inicio_s + v.muestra_inicio / fs
    v['segundo_en_fichero'] = segundo                                             # exacto: solo para comprobaciones
    v['minuto_en_fichero'] = np.floor(segundo / RESOLUCION_POSICION_S) * RESOLUCION_POSICION_S / 60
    v['duracion_fichero_min'] = v.edf_dur_s / 60
    v['fichero_con_crisis'] = (v.origen != 'fichero_sin_crisis').astype(int)
    return v.drop(columns=['montage', 'fs_original', 'mascara_canales', 'inicio_s', 'desviacion_mediana_uv', 'edf_dur_s'])
