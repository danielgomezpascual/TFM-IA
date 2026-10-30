"""
Cómo se guarda y se carga la señal preprocesada: un fichero .npy por tramo, en float16.

Motivo para cambiar: que cambie el formato o la organización de los ficheros de señal.

    RUTA_SENALES/<split>/<clase>/<id_tramo>.npy      matriz (22 canales x muestras), float16, en µV

float16 ocupa la mitad que float32. En µV y ya filtrada, la señal cabe de sobra (hasta ±65.504)
y el error de redondeo es de milésimas de µV (señal pequeña) a décimas (picos grandes), muy por
debajo del ruido del EEG. Se recorta a ±LIMITE_UV para que un artefacto enorme no se convierta en infinito.

Imagen: tfm_comun/imagenes/senales_float16.png (float32 frente a float16 y el error que se comete)
"""

import os

import numpy as np

TIPO_DATO = np.float16
LIMITE_UV = 60_000   # µV: por debajo del máximo de float16 (65.504)


def id_tramo(fichero, clase, inicio_s):
    """Nombre único y estable de un tramo, que sirve también de nombre de su fichero .npy.

    Qué hace:
        Une fichero, clase e inicio del tramo en milisegundos (9 cifras):
        'aaaaacrb_s001_t001_preictal_000123456'. No depende del orden de las filas de tramos.csv,
        así que no cambia si se vuelve a ejecutar 01_seleccion (y el preprocesado puede reanudarse).

    Parámetros:
        fichero: nombre del EDF sin extensión. clase: 'preictal', 'interictal' o 'ictal'.
        inicio_s: inicio del tramo en segundos desde el principio del fichero.

    Devuelve:
        El identificador (str).
    """
    return f'{fichero}_{clase}_{round(inicio_s * 1000):09d}'


def ruta_relativa(split, clase, identificador):
    """Dónde se guarda el .npy de un tramo, relativo a RUTA_SENALES: '<split>/<clase>/<identificador>.npy'.

    Usa '/' para que la misma ruta valga en Windows y en Linux (pathlib la entiende en los dos).
    Ejemplo: ruta_relativa('train', 'ictal', 'aaaaacrb_s001_t001_ictal_000097397') ->
             'train/ictal/aaaaacrb_s001_t001_ictal_000097397.npy'
    """
    return f'{split}/{clase}/{identificador}.npy'


def guardar_tramo(ruta, senal):
    """Guarda un tramo en disco como .npy en float16, de forma segura.

    Qué hace:
        1. Cuenta qué fracción de muestras pasa de ±LIMITE_UV (60.000 µV: casi seguro desconexiones).
        2. Recorta a ±LIMITE_UV (float16 no pasa de 65.504; por encima sería infinito).
        3. Convierte a float16 (la mitad de disco que float32, con error de centésimas de µV).
        4. Escribe en un fichero .tmp y luego lo renombra: si se corta la ejecución, nunca queda
           un .npy a medias (el renombrado es instantáneo).

    Parámetros:
        ruta: Path completo del .npy (crea las carpetas si no existen).
        senal: matriz (22 x muestras) en µV.

    Devuelve:
        Fracción de muestras recortadas (0.0 = ninguna). Se guarda en senales.csv.

    Imagen: tfm_comun/imagenes/senales_float16.png
    """
    senal = np.asarray(senal)
    recortadas = float(np.mean(np.abs(senal) > LIMITE_UV))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + '.tmp')
    with open(temporal, 'wb') as f:
        np.save(f, np.clip(senal, -LIMITE_UV, LIMITE_UV).astype(TIPO_DATO))
    os.replace(temporal, ruta)
    return recortadas


def cargar_tramo(ruta, en_memoria=False):
    """Carga el .npy de un tramo: matriz (22 x muestras) en float16 y µV.

    Qué hace:
        Por defecto lo abre con mmap (mapeado en memoria): no lee el fichero entero, solo los trozos
        que se usen. Así cortar una ventana de 4 s de un tramo de 2 horas es casi instantáneo.

    Parámetros:
        ruta: Path del .npy.
        en_memoria: True = lo lee entero a memoria (útil para comprobaciones o si se va a usar todo).

    Devuelve:
        Array (22 x muestras) en float16. Para operar con él, conviene pasarlo a float32.
    """
    return np.load(ruta, mmap_mode=None if en_memoria else 'r')


def bytes_tramo(n_canales, n_muestras):
    """Bytes que ocupará un tramo en disco: canales x muestras x 2 (float16). La cabecera del .npy aparte.

    Ejemplo: 22 canales x 1 hora a 256 Hz -> 22 x 921.600 x 2 ≈ 40,6 MB.
    """
    return n_canales * n_muestras * np.dtype(TIPO_DATO).itemsize
