"""
Bandas de frecuencia clásicas del EEG.

Motivo para cambiar: que cambien los límites de las bandas.
Sus colores están en colores.py (COLORES_BANDA).
"""

# Límites en Hz: [inferior, superior)
BANDAS_HZ = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alfa': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),   # hasta 45 Hz: por encima en EEG de superficie domina el músculo y la red (60 Hz)
}

# Con qué se asocia cada banda (para textos y tablas)
DESCRIPCION_BANDA = {
    'delta': 'ondas lentas: sueño profundo; en vigilia, lesión o actividad epiléptica',
    'theta': 'somnolencia; frecuente en niños',
    'alfa': 'reposo con ojos cerrados, sobre todo en la región occipital',
    'beta': 'alerta y concentración; aumenta con algunos fármacos',
    'gamma': 'procesamiento rápido; muy contaminada por actividad muscular',
}
