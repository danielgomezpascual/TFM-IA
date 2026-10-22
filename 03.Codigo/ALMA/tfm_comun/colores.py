"""
Colores de las figuras.

Motivo para cambiar: que cambien los colores de las figuras.
matplotlib no dibuja bien los emojis, así que en las figuras se usan colores
a juego con los iconos de iconos.py.
"""

# Particiones: azul 🔵, amarillo 🟡 y verde 🟢
COLORES_SPLIT = {'train': 'tab:blue', 'dev': 'goldenrod', 'eval': 'tab:green'}

# Etiquetas, a juego con sus iconos (🟥 fnsz rojo, 🟩 absz verde...)
COLORES_ETIQUETA = {
    'bckg': 'lightgray', 'seiz': 'black', 'sin_tipo': 'black',
    'fnsz': 'tab:red', 'gnsz': 'tab:blue', 'cpsz': 'tab:orange', 'absz': 'tab:green',
    'tcsz': 'tab:purple', 'tnsz': 'gold', 'spsz': 'tab:brown', 'mysz': 'dimgray',
}

# Periodos alrededor de una crisis (para predicción)
COLORES_PERIODO = {'interictal': 'tab:gray', 'preictal': 'tab:cyan', 'margen': 'lightgray',
                   'postictal': 'thistle'}

# Tipos de canal, a juego con sus iconos (🧠 eeg, 🫀 ecg...)
COLORES_TIPO_CANAL = {'eeg': 'tab:pink', 'ecg': 'tab:red', 'eog': 'tab:cyan', 'emg': 'tab:orange',
                      'stim': 'gold', 'resp': 'tab:blue', 'misc': 'tab:gray'}

# Regiones de la cabeza, a juego con sus corazones (💜 frontopolar, 💙 frontal...)
COLORES_REGION = {
    'frontopolar': 'tab:purple', 'frontal': 'tab:blue', 'central': 'tab:green',
    'temporal': 'tab:orange', 'parietal': 'gold', 'occipital': 'tab:red', 'auricular': 'tab:brown',
}

# Bandas de frecuencia: de la más lenta (morado oscuro) a la más rápida (amarillo)
COLORES_BANDA = {'delta': '#2a0593', 'theta': '#8f0da4', 'alfa': '#d04d73',
                 'beta': '#f58b47', 'gamma': '#e3b505'}
