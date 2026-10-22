"""
Lectura de ficheros EDF de TUSZ con MNE.

Motivo para cambiar: que cambie cómo se leen los EDF.
Qué es cada canal (tipo, región, referencia) no se decide aquí, sino en canales.py.
"""

import mne

from tfm_comun.canales import limpiar_nombre_canal, tipo_canal


def leer_edf_tusz(ruta_edf):
    """Lee un EDF de TUSZ y devuelve un mne.io.Raw con nombres y tipos de canal limpios.
    No añade anotaciones (eso es responsabilidad de anotaciones.py)."""
    raw = mne.io.read_raw_edf(ruta_edf, preload=True, verbose='error')
    raw.rename_channels(limpiar_nombre_canal)
    raw.set_channel_types({c: tipo_canal(c) for c in raw.ch_names}, on_unit_change='ignore',
                          verbose='error')
    return raw
