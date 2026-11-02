"""
Catálogo de regiones del cerebro: a qué giro y a qué lóbulo pertenece cada zona de la corteza.

Motivo para cambiar: que cambie el atlas o cómo se agrupan sus regiones en lóbulos.
Es el equivalente, dentro del cerebro, de lo que REGION_ELECTRODO (canales.py) es en el cuero
cabelludo: allí se dice en qué región de la cabeza está cada electrodo; aquí, en qué región de
la corteza está cada punto del cerebro plantilla (fsaverage).

Atlas: Desikan-Killiany ('aparc' en FreeSurfer) [Desikan et al., 2006]. Divide cada hemisferio
en 34 regiones siguiendo los giros. Sus nombres vienen en inglés y sin espacios
('superiortemporal-rh'); aquí se traducen y se agrupan en lóbulos.
"""

ATLAS = 'aparc'

# Región del atlas -> (nombre en español, lóbulo)
REGIONES_APARC = {
    # Frontal
    'superiorfrontal': ('giro frontal superior', 'frontal'),
    'rostralmiddlefrontal': ('giro frontal medio (anterior)', 'frontal'),
    'caudalmiddlefrontal': ('giro frontal medio (posterior)', 'frontal'),
    'parsopercularis': ('pars opercularis (frontal inferior)', 'frontal'),
    'parstriangularis': ('pars triangularis (frontal inferior)', 'frontal'),
    'parsorbitalis': ('pars orbitalis (frontal inferior)', 'frontal'),
    'lateralorbitofrontal': ('corteza orbitofrontal lateral', 'frontal'),
    'medialorbitofrontal': ('corteza orbitofrontal medial', 'frontal'),
    'precentral': ('giro precentral (motor)', 'frontal'),
    'paracentral': ('lobulillo paracentral', 'frontal'),
    'frontalpole': ('polo frontal', 'frontal'),
    # Parietal
    'superiorparietal': ('lobulillo parietal superior', 'parietal'),
    'inferiorparietal': ('lobulillo parietal inferior', 'parietal'),
    'supramarginal': ('giro supramarginal', 'parietal'),
    'postcentral': ('giro postcentral (sensitivo)', 'parietal'),
    'precuneus': ('precúneo', 'parietal'),
    # Temporal
    'superiortemporal': ('giro temporal superior', 'temporal'),
    'middletemporal': ('giro temporal medio', 'temporal'),
    'inferiortemporal': ('giro temporal inferior', 'temporal'),
    'bankssts': ('surco temporal superior', 'temporal'),
    'fusiform': ('giro fusiforme', 'temporal'),
    'transversetemporal': ('giro temporal transverso (auditivo)', 'temporal'),
    'entorhinal': ('corteza entorrinal', 'temporal'),
    'temporalpole': ('polo temporal', 'temporal'),
    'parahippocampal': ('giro parahipocampal', 'temporal'),
    # Occipital
    'lateraloccipital': ('corteza occipital lateral', 'occipital'),
    'lingual': ('giro lingual', 'occipital'),
    'cuneus': ('cúneo', 'occipital'),
    'pericalcarine': ('corteza pericalcarina (visual primaria)', 'occipital'),
    # Cíngulo e ínsula
    'rostralanteriorcingulate': ('cíngulo anterior (rostral)', 'cíngulo'),
    'caudalanteriorcingulate': ('cíngulo anterior (caudal)', 'cíngulo'),
    'posteriorcingulate': ('cíngulo posterior', 'cíngulo'),
    'isthmuscingulate': ('istmo del cíngulo', 'cíngulo'),
    'insula': ('ínsula', 'ínsula'),
}

LOBULOS = ['frontal', 'temporal', 'parietal', 'occipital', 'cíngulo', 'ínsula']

# Sufijo de hemisferio de FreeSurfer -> nombre en español
HEMISFERIOS = {'lh': 'izquierdo', 'rh': 'derecho'}


def describir_region(nombre_etiqueta):
    """Nombre en español, lóbulo y hemisferio de una región del atlas.

    Ejemplo: 'superiortemporal-rh' -> ('giro temporal superior', 'temporal', 'derecho').
    Las zonas sin región ('unknown', 'corpuscallosum') devuelven (None, None, hemisferio).
    """
    region, hemi = nombre_etiqueta.rsplit('-', 1)
    nombre, lobulo = REGIONES_APARC.get(region, (None, None))
    return nombre, lobulo, HEMISFERIOS[hemi]
