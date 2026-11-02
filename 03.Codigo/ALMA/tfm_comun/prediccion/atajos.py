"""
Atajos: pistas que no son el EEG preictal pero que podrían delatar la clase de una ventana.

Motivo para cambiar: que cambien las pistas que se vigilan o cómo se miden y se controlan.
Lo usa fase_02/05_auditoria_atajos (y servirá en la Fase 3 para el experimento de control).

Qué es un atajo (shortcut learning, Geirhos et al. 2020)
=========================================================
Una red aprende lo más fácil que separe las clases, sea o no lo que buscamos. Si las ventanas
⏳ preictales salen casi siempre de un tipo de equipo, del principio de la grabación o con dos canales
a 0, la red puede acertar mirando ESO y no los cambios del EEG antes de una crisis. En eval saldría una
buena nota que no se sostiene con otros datos (como los modelos de radiografía que reconocían el
hospital por una marca en la placa, Zech et al. 2018).

Cómo se mide
============
Cada pista se convierte en un número por ventana (una "variable de atajo") y se mira cuánto separa
las clases con la AUROC: 0,5 = no separa nada; 1 (o 0) = separa perfectamente. Como una variable
puede separar "al revés" (p. ej. las ⏳ preictales salen ANTES en el fichero, AUROC < 0,5), se usa la
FUERZA = max(AUROC, 1 - AUROC), que va de 0,5 (nada) a 1 (todo), y aparte la dirección.

Como una pista puede separar con "forma" (p. ej. el primer minuto del fichero es casi solo 💤 y los minutos 2-5
casi solo ⏳), la fuerza se calcula sobre la variable CODIFICADA POR FRANJAS (codificar_por_franjas): cada franja
se sustituye por su proporción de ⏳ en train. Así se captan también relaciones que no son "más / menos".

Todo trabaja con tablas (sin leer la señal): es rápido y no depende del modelo que se entrene después.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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

# Franjas para la codificación no monótona de las variables numéricas
N_FRANJAS = 10

# ── Semáforo de la fuerza ──────────────────────────────────────────────────────────────────────────
# No hay un umbral "oficial": son cortes habituales para leer una AUROC. Por debajo de 0,55 la variable
# casi no separa; por encima de 0,65 separa lo bastante como para que una red la aproveche.
FUERZA_DEBIL = 0.55
FUERZA_FUERTE = 0.65


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


def codificar_por_tasa(categorias_ajuste, etiqueta_ajuste, categorias):
    """Convierte una variable de categorías (p. ej. el montaje) en un número: la proporción de ⏳ preictal
    que tiene cada categoría en los datos de ajuste (normalmente 🔵 train).

    Para qué: la AUROC necesita un número que ordene las ventanas. Con categorías no hay orden natural;
    así se ordenan de "la categoría más preictal" a "la menos". Se ajusta con train y se aplica igual a
    dev y eval (si no, se haría trampa con sus etiquetas). Una categoría que no estaba en train recibe la
    proporción global de train.
    Ejemplo: en train, 03_tcp_ar_a tiene 60 % de ⏳ preictal y 01_tcp_ar 45 % -> 03_tcp_ar_a = 0,60; 01_tcp_ar = 0,45.
    """
    etiqueta_ajuste = pd.Series(np.asarray(etiqueta_ajuste))
    tasa = etiqueta_ajuste.groupby(np.asarray(categorias_ajuste)).mean()
    return pd.Series(np.asarray(categorias)).map(tasa).fillna(etiqueta_ajuste.mean()).to_numpy()


def codificar_por_franjas(valores_ajuste, etiqueta_ajuste, valores, n_franjas=N_FRANJAS):
    """Convierte una variable numérica en "proporción de ⏳ preictal de su franja", ajustada con los datos de ajuste.

    Para qué: la AUROC de la variable tal cual solo ve relaciones de "más / menos". Si la pista tiene forma
    (p. ej. el minuto 0-1 es casi solo 💤, los minutos 2-5 casi solo ⏳ y el resto mezclado), la AUROC tal cual
    sale cerca de 0,5 aunque un modelo sí podría usarla. Con esta codificación también se capta.
    Qué hace:
        1. Si la variable tiene n_franjas valores distintos o menos (p. ej. 0/1, o 0/2 canales a cero), cada valor es
           su propia franja. Si no, se corta en n_franjas franjas con el mismo número de ventanas (cuantiles de los
           datos de ajuste; con valores muy repetidos salen menos).
        2. Cada franja vale la proporción de ⏳ preictal que tiene en los datos de ajuste.
        3. Se aplica igual a los datos que se quieren medir (dev y eval no usan sus etiquetas).
    Devuelve: array con un número por ventana de `valores`.
    """
    ajuste = np.asarray(valores_ajuste, dtype=float)
    if len(np.unique(ajuste)) <= n_franjas:                                       # pocos valores: uno por franja
        return codificar_por_tasa(ajuste, etiqueta_ajuste, np.asarray(valores, dtype=float))
    interiores = np.unique(np.quantile(ajuste, np.linspace(0, 1, n_franjas + 1)[1:-1]))   # bordes entre franjas
    franja = lambda x: np.searchsorted(interiores, np.asarray(x, dtype=float), side='right')
    return codificar_por_tasa(franja(ajuste), etiqueta_ajuste, franja(valores))


def marcar_sesiones(ventanas, edfs):
    """Añade a cada ventana su sesión y si es de una sesión con crisis, para el control de "tipo de fichero".

    Por qué: el 💤 interictal sale casi todo de ficheros sin crisis y el ⏳ preictal de ficheros con crisis. Para
    saber si la red aprende "día con crisis", hace falta 💤 de un día con crisis pero lejos de ellas. El de
    ficheros con crisis ('lejos_de_crisis') casi siempre está JUSTO DESPUÉS de la crisis (EEG aún alterado), así
    que se usa otro: 💤 de ficheros SIN crisis de la MISMA SESIÓN (mismo día) que sí tiene crisis.
    Columnas que añade:
        sesion                     'aaaaaacz_s006'
        sesion_con_crisis          True si algún fichero de la sesión tiene crisis (edfs.n_seiz_events_bi > 0)
        antes_de_la_primera_crisis True si el fichero es un token ANTERIOR al primer fichero con crisis de su sesión
                                   (variante estricta: no puede ser post-crisis; ojo, no se sabe si los tokens
                                   son seguidos ni cuánto tiempo hay entre ellos)
    """
    e = edfs[['file', 'n_seiz_events_bi']].copy()
    e['sesion'] = e.file.str.rsplit('_', n=1).str[0]
    e['token'] = e.file.str.rsplit('_', n=1).str[1].str[1:].astype(int)
    primera = e[e.n_seiz_events_bi > 0].groupby('sesion').token.min()
    v = ventanas.copy()
    v['sesion'] = v.file.str.rsplit('_', n=1).str[0]
    token = v.file.str.rsplit('_', n=1).str[1].str[1:].astype(int)
    v['sesion_con_crisis'] = v.sesion.isin(primera.index)
    v['antes_de_la_primera_crisis'] = v.sesion_con_crisis & (token < v.sesion.map(primera))
    return v


def fuerza_auroc(etiqueta, valores):
    """AUROC de una variable, su fuerza y su dirección.

    Parámetros:
        etiqueta: 1 = ⏳ preictal, 0 = 💤 interictal.  valores: la variable (un número por ventana).
    Devuelve:
        (auroc, fuerza, direccion):
            auroc: probabilidad de que una ⏳ preictal al azar tenga un valor MAYOR que una 💤 interictal al azar.
            fuerza: max(auroc, 1 - auroc), de 0,5 (no separa) a 1 (separa todo).
            direccion: 'más alta en ⏳ preictal', 'más baja en ⏳ preictal' o 'igual'.
        Si solo hay una clase o la variable es constante, (0,5, 0,5, 'igual').
    Ejemplo: minuto_en_fichero con AUROC 0,30 -> fuerza 0,70, 'más baja en ⏳ preictal' (salen antes en el fichero).
    """
    etiqueta, valores = np.asarray(etiqueta), np.asarray(valores, dtype=float)
    if len(np.unique(etiqueta)) < 2 or np.nanstd(valores) == 0:
        return 0.5, 0.5, 'igual'
    auroc = float(roc_auc_score(etiqueta, valores))
    direccion = 'igual' if abs(auroc - 0.5) < 0.005 else ('más alta en ⏳ preictal' if auroc > 0.5 else 'más baja en ⏳ preictal')
    return auroc, max(auroc, 1 - auroc), direccion


def fuerza_dentro_del_paciente(ventanas, variable, columna_paciente='patient', columna_etiqueta='etiqueta'):
    """Fuerza de una variable DENTRO de cada paciente, promediada (pesando por ventanas).

    Por qué importa: el entrenamiento está equilibrado paciente a paciente (balanceo.py), así que una
    pista que solo cambia ENTRE pacientes (p. ej. "este paciente se grabó con un equipo de 250 Hz") no
    le sirve a la red: cada paciente aporta mitad y mitad. La pista peligrosa es la que cambia DENTRO
    del paciente: sus ⏳ preictales y sus 💤 interictales salen de situaciones distintas.
    Para cada paciente con las dos clases se calcula la AUROC; se promedian y se devuelve la fuerza de ese
    promedio. Un paciente con la variable constante cuenta como 0,5 (dentro de él, no separa).

    Devuelve:
        (auroc_media, fuerza, n_pacientes).
    """
    aurocs, pesos = [], []
    for _, v in ventanas.groupby(columna_paciente):
        if v[columna_etiqueta].nunique() < 2:
            continue
        aurocs.append(fuerza_auroc(v[columna_etiqueta], v[variable])[0])
        pesos.append(len(v))
    if not aurocs:
        return 0.5, 0.5, 0
    media = float(np.average(aurocs, weights=pesos))
    return media, max(media, 1 - media), len(aurocs)


def semaforo(fuerza):
    """🟢 / 🟡 / 🔴 según la fuerza de una variable: < FUERZA_DEBIL (0,55) no separa, hasta FUERZA_FUERTE (0,65)
    separa un poco, por encima separa lo bastante como para que una red la use."""
    if fuerza < FUERZA_DEBIL:
        return '🟢 no separa'
    if fuerza < FUERZA_FUERTE:
        return '🟡 separa un poco'
    return '🔴 separa mucho'


def emparejar_por_posicion(ventanas, ancho_min=1.0, ratio=1.0, semilla=42,
                           columna_paciente='patient', columna_minuto='minuto_en_fichero'):
    """Máscara de un conjunto de CONTROL en el que las 💤 interictales salen de los mismos minutos del
    fichero que las ⏳ preictales, paciente a paciente.

    Qué hace:
        1. Parte el fichero en franjas de `ancho_min` minutos (0-1, 1-2, 2-3...).
        2. Para cada paciente y franja, cuenta sus ⏳ preictales (n) y sortea round(ratio x n) 💤 interictales
           SUYAS de esa misma franja (todas las que haya, si tiene menos).
        3. Se quedan todas las ⏳ preictales y las 💤 interictales sorteadas. Los pacientes sin ⏳ preictales no aportan.
        Así, si la red acierta solo porque "las ⏳ preictales están al principio del fichero", en este conjunto
        ya no le sirve: las dos clases tienen la misma posición. Si su nota baja mucho aquí, usaba el atajo.

    Parámetros:
        ventanas: tabla con 'clase', paciente y minuto_en_fichero (de metadatos_de_ventanas).
        ancho_min: anchura de las franjas (min). Más estrecha = emparejado más fino, pero menos ventanas.
        ratio: 💤 interictales por cada ⏳ preictal en cada franja.
        semilla: para que el sorteo salga siempre igual.

    Devuelve:
        (mascara, cumplido): Serie verdadero/falso con el índice de `ventanas`, y la fracción de 💤 interictales
        pedidas que se han podido conseguir (1 = todas; menos de 1 = en algunas franjas faltaban).
    Ejemplo: un paciente con 30 ⏳ preictales entre los minutos 3 y 4 y 200 💤 interictales en esa franja ->
        se sortean 30 de esas 200 (con ratio 1).
    """
    rng = np.random.default_rng(semilla)
    franja = np.floor(ventanas[columna_minuto] / ancho_min).astype(int)
    clave = [ventanas[columna_paciente], franja]
    n_pre = ventanas.clase.eq('preictal').groupby(clave).sum()                     # ⏳ preictales por (paciente, franja)
    candidatas = ventanas.index[ventanas.clase == 'interictal'].to_series() \
        .groupby([c[ventanas.clase == 'interictal'] for c in clave]).apply(lambda s: np.sort(s.to_numpy()))
    elegidas, pedidas, conseguidas = [], 0, 0
    for (paciente, f), n in n_pre[n_pre > 0].sort_index().items():               # orden fijo: el sorteo no depende del orden de la tabla
        disponibles = candidatas.get((paciente, f), np.array([], dtype=int))
        objetivo = int(round(n * ratio))
        k = min(len(disponibles), objetivo)
        pedidas += objetivo
        conseguidas += k
        if k:
            elegidas.extend(rng.choice(disponibles, size=k, replace=False))
    mascara = (ventanas.clase == 'preictal') | ventanas.index.isin(elegidas)
    return mascara, (conseguidas / pedidas if pedidas else np.nan)
