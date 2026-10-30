"""
Montaje bipolar TCP (Temporal Central Parasagittal) de 22 canales.

Motivo para cambiar: que cambie el montaje con el que se trabaja.

- montaje_tcp(): con MNE, para dibujar (los canales que se puedan construir).
- senal_tcp() / tcp_desde_referencial(): matriz de 22 canales siempre, para el preprocesado y la demo.

Imagen: tfm_comun/imagenes/montaje_tcp_desde_referencial.png (las 5 cadenas y cómo la resta quita la referencia)
"""

import mne
import numpy as np

# Las 5 cadenas del montaje: pares (ánodo, cátodo) en el orden de TUSZ.
# Cada canal TCP es la resta ánodo - cátodo de dos electrodos vecinos.
CADENAS_TCP = {
    'temporal izquierda': [('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1')],
    'temporal derecha': [('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2')],
    'central': [('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'), ('T4', 'A2')],
    'parasagital izquierda': [('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1')],
    'parasagital derecha': [('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')],
}
PARES_TCP = [par for pares in CADENAS_TCP.values() for par in pares]
CANALES_TCP = [f'{a}-{c}' for a, c in PARES_TCP]

# Canales que se dibujan en las figuras: uno de cada cadena, para ver toda la cabeza sin dibujar los 22.
# Para usar otros, se cambian aquí (deben estar en CANALES_TCP) y cambian en todos los notebooks a la vez.
CANALES_FIGURAS = ['FP1-F7',   # temporal izquierda
                   'FP2-F8',   # temporal derecha
                   'C3-CZ',    # central
                   'P3-O1',    # parasagital izquierda
                   'F4-C4']    # parasagital derecha
assert set(CANALES_FIGURAS) <= set(CANALES_TCP), 'CANALES_FIGURAS tiene un canal que no es del montaje TCP.'


def montaje_tcp(raw):
    """Convierte un Raw referencial de MNE en un Raw bipolar TCP, para DIBUJAR la señal.

    Qué hace:
        Crea cada canal TCP como la resta de dos electrodos vecinos (p. ej. FP1 - F7) con
        mne.set_bipolar_reference y deja solo esos canales, en el orden de TUSZ.
        A diferencia de senal_tcp(), no rellena con ceros: si al fichero le faltan A1/A2
        (montaje 03_tcp_ar_a), devuelve solo los 20 canales que se pueden construir.

    Parámetros:
        raw: mne.io.Raw con los electrodos referenciales y nombres limpios ('FP1', 'A1'...), de edf.py.

    Devuelve:
        mne.io.Raw con 22 (o 20) canales llamados 'FP1-F7', 'F7-T3'...

    Imagen: tfm_comun/imagenes/montaje_tcp_desde_referencial.png
    """
    pares = [(a, c) for a, c in PARES_TCP if a in raw.ch_names and c in raw.ch_names]
    tcp = mne.set_bipolar_reference(raw.copy().pick('eeg'), anode=[a for a, _ in pares],
                                    cathode=[c for _, c in pares], drop_refs=True, verbose='error')
    return tcp.pick([f'{a}-{c}' for a, c in pares])


def tcp_desde_referencial(datos, nombres):
    """Construye el montaje TCP de 22 canales a partir de una matriz de electrodos (sin MNE).

    Qué hace:
        Para cada par (ánodo, cátodo) de PARES_TCP resta las dos filas: canal = ánodo - cátodo.
        Al restar, todo lo que tienen en común los dos electrodos (la referencia, que es lo que más
        cambia de un equipo a otro) se cancela. Si falta alguno de los dos electrodos (p. ej. A1/A2
        en el montaje 03_tcp_ar_a), el canal se deja a 0 y se marca como no válido. Así TODAS las
        señales tienen la misma forma (22 x muestras), que es lo que necesita el modelo.

    Parámetros:
        datos: matriz (electrodos x muestras), en V o en µV (la resta conserva la unidad).
        nombres: nombre limpio de cada fila de datos ('FP1', 'A1'...), en el mismo orden.

    Devuelve:
        (tcp, canales_validos):
            tcp: matriz (22 x muestras) en el orden de CANALES_TCP.
            canales_validos: array de 22 booleanos; False = canal a 0 porque faltaba un electrodo.

    Ejemplo:
        tcp, validos = tcp_desde_referencial(datos_uv, ['FP1', 'F7', 'T3', ...])

    Imagen: tfm_comun/imagenes/montaje_tcp_desde_referencial.png
    """
    datos = np.asarray(datos)
    posicion = {n: i for i, n in enumerate(nombres)}
    tcp = np.zeros((len(PARES_TCP), datos.shape[-1]), dtype=datos.dtype)
    validos = np.zeros(len(PARES_TCP), dtype=bool)
    for k, (anodo, catodo) in enumerate(PARES_TCP):
        if anodo in posicion and catodo in posicion:
            tcp[k] = datos[posicion[anodo]] - datos[posicion[catodo]]
            validos[k] = True
    return tcp, validos


def senal_tcp(raw):
    """Montaje TCP de 22 canales a partir de un Raw de MNE (para el preprocesado del TFM).

    Qué hace:
        Saca del Raw solo los electrodos que usa el montaje TCP y llama a tcp_desde_referencial().
        Es como montaje_tcp(), pero devuelve SIEMPRE los 22 canales (a 0 los que no se pueden
        construir), para que todas las señales tengan la misma forma.

    Parámetros:
        raw: mne.io.Raw referencial con nombres limpios, de edf.py.

    Devuelve:
        (tcp, canales_validos): matriz (22 x muestras) en V y array de 22 booleanos.

    Imagen: tfm_comun/imagenes/montaje_tcp_desde_referencial.png
    """
    electrodos = sorted({e for par in PARES_TCP for e in par} & set(raw.ch_names))
    return tcp_desde_referencial(raw.get_data(picks=electrodos), electrodos)
