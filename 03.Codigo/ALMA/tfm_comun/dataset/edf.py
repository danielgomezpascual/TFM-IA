"""
Lectura de ficheros EDF de TUSZ con MNE.

Motivo para cambiar: que cambie cómo se leen los EDF.
Qué es cada canal (tipo, región, referencia) no se decide aquí, sino en canales.py.
"""

import mne

from tfm_comun.dataset.canales import limpiar_nombre_canal, tipo_canal


def leer_edf_tusz(ruta_edf, inicio_s=None, fin_s=None):
    """Lee un EDF de TUSZ (entero o solo un trozo) y lo deja listo para trabajar.

    Qué hace:
        1. Abre el EDF con MNE sin cargar la señal (preload=False: solo lee la cabecera).
        2. Si se pide un trozo, lo recorta ANTES de cargar: solo se lee del disco lo necesario
           (mucho más rápido en ficheros largos).
        3. Carga la señal, limpia los nombres de canal ('EEG FP1-LE' -> 'FP1') y pone a cada canal
           su tipo (eeg, ecg, eog...), con las funciones de canales.py.
        No añade anotaciones: eso es cosa de anotaciones.py.

    Parámetros:
        ruta_edf: ruta del .edf.
        inicio_s, fin_s: trozo a leer, en segundos desde el inicio del fichero (None = desde el
            principio / hasta el final). Se ajustan a los límites del fichero.

    Devuelve:
        mne.io.Raw con la señal en voltios. Si se leyó un trozo, raw.first_samp / raw.info['sfreq']
        dice en qué segundo del fichero empieza lo leído.
    """
    raw = mne.io.read_raw_edf(ruta_edf, preload=False, verbose='error')
    if inicio_s is not None or fin_s is not None:
        inicio = max(0.0, inicio_s or 0.0)
        fin = min(raw.times[-1], fin_s if fin_s is not None else raw.times[-1])
        raw.crop(tmin=inicio, tmax=fin)
    raw.load_data(verbose='error')
    raw.rename_channels(limpiar_nombre_canal)
    raw.set_channel_types({c: tipo_canal(c) for c in raw.ch_names}, on_unit_change='ignore',
                          verbose='error')
    return raw
