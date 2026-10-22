"""
Montaje bipolar TCP (Temporal Central Parasagittal) de 22 canales.

Motivo para cambiar: que cambie el montaje con el que se trabaja.
"""

import mne

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


def montaje_tcp(raw):
    """Raw referencial (LE o AR) -> Raw bipolar TCP de 22 canales, en el orden de TUSZ.
    Si al fichero le faltan A1/A2 (montaje 03_tcp_ar_a), se queda con los 20 que se pueden construir."""
    pares = [(a, c) for a, c in PARES_TCP if a in raw.ch_names and c in raw.ch_names]
    tcp = mne.set_bipolar_reference(raw.copy().pick('eeg'), anode=[a for a, _ in pares],
                                    cathode=[c for _, c in pares], drop_refs=True, verbose='error')
    return tcp.pick([f'{a}-{c}' for a, c in pares])
