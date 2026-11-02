"""
Localización de fuentes: estimar DÓNDE, dentro del cerebro, se origina la actividad que miden los electrodos.

Motivo para cambiar: que cambie cómo se estima el origen de la crisis (cerebro plantilla, modelo de
la cabeza, método inverso o medida que se compara).

Idea en tres pasos [Michel et al., 2004]:
    1. Problema directo: con un modelo de la cabeza (cerebro, cráneo y cuero cabelludo) se calcula
       qué vería cada electrodo si se activara cada punto de la corteza (modelo directo, "forward").
    2. Problema inverso: se recorre el camino al revés. A partir de lo que miden los electrodos se
       estima la actividad de cada punto de la corteza. Usamos sLORETA [Pascual-Marqui, 2002], que
       da la solución más suave posible y con poco error de localización.
    3. Comparación: potencia de cada punto al INICIO de la crisis dividida por su potencia en un
       tramo basal (interictal). Donde más crece es el origen estimado.

Limitaciones (hay que decirlas siempre que se enseñe la figura):
    - Cerebro PLANTILLA (fsaverage, la media de 40 personas [Fischl et al., 1999]): TUSZ no trae la
      resonancia de cada paciente.
    - Solo unos 19 electrodos: la precisión es de LÓBULO, no de centímetros [Lantz et al., 2003].
    Es una visualización orientativa, no un diagnóstico.

Necesita el cerebro plantilla de MNE (fetch_fsaverage lo descarga la primera vez, unos 200 MB, en
la carpeta mne_data del usuario).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_cov

from tfm_comun.dataset.canales import region_electrodo
from tfm_comun.dataset.regiones_cerebro import ATLAS, describir_region

# Posiciones 3D estándar de los electrodos 10-20 ('standard_1020' pasa a llamarse así en MNE 1.13).
# Incluye los nombres antiguos de TUSZ (T3, T4, T5, T6)
MONTAJE_ESTANDAR = 'colin27_1020'

# Orejas: con referencia de orejas unidas (-LE) se miden casi contra sí mismas y no aportan
ELECTRODOS_EXCLUIDOS = ('A1', 'A2')

BANDA_FUENTES_HZ = (1, 30)   # quita derivas lentas y músculo; deja dentro la firma de todos los tipos
METODO_INVERSO = 'sLORETA'
SNR = 3                      # relación señal/ruido supuesta: regularización lambda² = 1 / SNR²


def cerebro_plantilla():
    """Rutas del cerebro plantilla fsaverage: carpeta de sujetos, fuentes y modelo de la cabeza.

    La primera vez lo descarga (mne.datasets.fetch_fsaverage); después lo lee del disco.
    Devuelve un diccionario con:
        subjects_dir: carpeta que contiene 'fsaverage' (la piden las figuras 3D).
        fuentes: los 20.484 puntos de la corteza donde se estima la actividad (ico-5, ~5 mm entre puntos).
        bem: modelo de la cabeza en 3 capas (cerebro, cráneo, cuero cabelludo).
    """
    ruta = Path(mne.datasets.fetch_fsaverage(verbose='error'))
    return {'subjects_dir': ruta.parent,
            'fuentes': ruta / 'bem' / 'fsaverage-ico-5-src.fif',
            'bem': ruta / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'}


def electrodos_con_posicion(raw, excluir=()):
    """Copia del Raw con solo los electrodos 10-20 que tienen posición 3D estándar, ya colocada.

    Qué hace:
        1. Se queda con los electrodos de cuero cabelludo (los que tienen región en canales.py).
        2. Quita las orejas (ELECTRODOS_EXCLUIDOS) y los que se pidan en excluir (p. ej. ruidosos).
        3. Quita los que no están en el montaje estándar (T1, T2) y coloca el resto en la cabeza.
    Las mayúsculas no importan: 'FP1' de TUSZ es 'Fp1' en el montaje.
    """
    montaje = mne.channels.make_standard_montage(MONTAJE_ESTANDAR)
    con_posicion = {c.upper() for c in montaje.ch_names}
    excluidos = set(ELECTRODOS_EXCLUIDOS) | set(excluir)
    electrodos = [c for c in raw.ch_names
                  if region_electrodo(c) and c not in excluidos and c.upper() in con_posicion]
    eeg = raw.copy().pick(electrodos)
    eeg.set_montage(montaje, match_case=False, verbose='error')
    return eeg


def modelo_directo(info, plantilla, carpeta=None):
    """Modelo directo (qué vería cada electrodo si se activara cada punto de la corteza), guardado en disco.

    Es la parte lenta (de medio minuto a un par de minutos) y solo depende de QUÉ electrodos hay, no de la
    señal: con el cerebro plantilla, dos ficheros con los mismos electrodos tienen el mismo modelo.
    Por eso se guarda la primera vez en carpeta/directo_<n electrodos>_<huella>-fwd.fif y las siguientes
    se lee (unos segundos). La huella es un resumen corto de la lista de electrodos.

    Parámetros:
        info: info de MNE con los electrodos ya colocados (electrodos_con_posicion).
        plantilla: el diccionario de cerebro_plantilla().
        carpeta: dónde guardarlo (None = no guardar, calcular siempre).
    """
    electrodos = sorted(info.ch_names)
    huella = hashlib.md5('_'.join(electrodos).encode()).hexdigest()[:8]
    ruta = Path(carpeta) / f'directo_{len(electrodos)}_{huella}-fwd.fif' if carpeta else None
    if ruta is not None and ruta.exists():
        return mne.read_forward_solution(ruta, verbose='error')
    directo = mne.make_forward_solution(info, trans='fsaverage', src=str(plantilla['fuentes']),
                                        bem=str(plantilla['bem']), eeg=True, meg=False, mindist=5.0,
                                        verbose='error')
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        mne.write_forward_solution(ruta, directo, overwrite=True, verbose='error')
    return directo


def _potencia_media(datos):
    """Potencia media (media del cuadrado) de cada fila de una matriz (filas × muestras)."""
    return (datos ** 2).mean(axis=1)


def origen_crisis(raw, tramo_basal, tramo_inicio, excluir=(), banda_hz=BANDA_FUENTES_HZ, notch_hz=60,
                  carpeta_modelos=None):
    """Estima dónde se origina la crisis: cuánto crece la potencia en cada electrodo y en cada punto
    de la corteza al inicio de la crisis respecto a un tramo basal.

    Parámetros:
        raw: Raw REFERENCIAL de TUSZ (el de leer_edf_tusz, sin montaje TCP: aquí hace falta un
            valor por electrodo, no restas de vecinos).
        tramo_basal: (inicio, fin) en segundos de señal normal (interictal). Cuanto más largo, mejor
            se estima el ruido (≥ 30 s recomendable).
        tramo_inicio: (inicio, fin) en segundos de los primeros segundos de la crisis.
        excluir: electrodos que no se usan (p. ej. los ruidosos).
        banda_hz, notch_hz: filtrado antes de calcular.
        carpeta_modelos: dónde se guardan los modelos directos para reutilizarlos (ver modelo_directo).

    Qué hace:
        1. Electrodos con posición 3D, filtrados, con referencia promedio (la que pide el método inverso).
        2. Modelo directo sobre fsaverage y covarianza del ruido con el tramo basal.
        3. sLORETA sobre la covarianza de cada tramo (rápido y con poca memoria) y cociente de
           potencias punto a punto, en decibelios
           (+10 dB = 10 veces más potencia que en el basal; 0 dB = sin cambio).
        4. Lo mismo electrodo a electrodo, para el mapa del cuero cabelludo.

    Devuelve un diccionario con:
        cociente_db: array con el aumento (dB) en cada punto de la corteza (primero el hemisferio
            izquierdo y luego el derecho).
        vertices: [vértices izquierdos, vértices derechos] de esos puntos en fsaverage.
        electrodos_db: pd.Series con el aumento (dB) en cada electrodo.
        info: info de MNE de los electrodos usados (con posiciones, para el topomap).
        subjects_dir: carpeta de fsaverage (para las figuras 3D).
    """
    plantilla = cerebro_plantilla()
    eeg = electrodos_con_posicion(raw, excluir)
    eeg.notch_filter(notch_hz, verbose='error').filter(*banda_hz, verbose='error')
    eeg.set_eeg_reference('average', projection=True, verbose='error')

    # 1. Modelo directo: qué vería cada electrodo si se activara cada punto de la corteza
    directo = modelo_directo(eeg.info, plantilla, carpeta_modelos)
    # 2. Ruido: covarianza entre electrodos en el tramo basal
    ruido = mne.compute_raw_covariance(eeg, tmin=tramo_basal[0], tmax=tramo_basal[1], method='empirical',
                                       verbose='error')
    inverso = make_inverse_operator(eeg.info, directo, ruido, loose=0.2, depth=0.8, verbose='error')

    # 3. sLORETA en cada tramo y cociente de potencias
    # Se aplica a la COVARIANZA de cada tramo, no muestra a muestra: da directamente la potencia de cada
    # punto. Muestra a muestra serían ~20.000 puntos × decenas de miles de muestras: varios GB de memoria
    def potencia_fuentes(tramo):
        covarianza = mne.compute_raw_covariance(eeg, tmin=tramo[0], tmax=tramo[1], method='empirical',
                                                verbose='error')
        stc = apply_inverse_cov(covarianza, eeg.info, inverso, nave=1, lambda2=1 / SNR ** 2,
                                method=METODO_INVERSO, verbose='error')
        return stc.data[:, 0], stc.vertices

    potencia_basal, vertices = potencia_fuentes(tramo_basal)
    potencia_inicio, _ = potencia_fuentes(tramo_inicio)
    cociente_db = 10 * np.log10(potencia_inicio / potencia_basal)

    # 4. Lo mismo en cada electrodo (con la referencia promedio aplicada)
    datos = eeg.copy().apply_proj(verbose='error').get_data()
    def potencia_electrodos(tramo):
        inicio, fin = eeg.time_as_index(list(tramo))
        return _potencia_media(datos[:, inicio:fin])
    electrodos_db = pd.Series(10 * np.log10(potencia_electrodos(tramo_inicio) / potencia_electrodos(tramo_basal)),
                              index=eeg.ch_names, name='aumento (dB)')

    return {'cociente_db': cociente_db, 'vertices': vertices, 'electrodos_db': electrodos_db,
            'info': eeg.info, 'subjects_dir': plantilla['subjects_dir']}


def resumen_por_region(cociente_db, vertices, subjects_dir, umbral_db):
    """Traduce el mapa de la corteza a regiones con nombre: dónde está el máximo y cómo se reparte por lóbulos.

    Parámetros:
        cociente_db, vertices, subjects_dir: los de origen_crisis().
        umbral_db: a partir de qué aumento un punto cuenta como "activo" (el mismo que se pinta en la figura).

    Devuelve:
        maximo: diccionario con la región, el lóbulo, el hemisferio y el aumento (dB) del punto que más crece.
        lobulos: DataFrame con una fila por lóbulo y hemisferio: aumento medio (dB), aumento máximo (dB)
            y % de sus puntos por encima del umbral. Ordenado de más a menos aumento medio.
    """
    etiquetas = mne.read_labels_from_annot('fsaverage', ATLAS, subjects_dir=subjects_dir, verbose='error')
    n_izquierdo = len(vertices[0])
    filas, region_de = [], np.full(len(cociente_db), None, dtype=object)
    for etiqueta in etiquetas:
        nombre, lobulo, hemisferio = describir_region(etiqueta.name)
        if lobulo is None:
            continue
        lado = 0 if etiqueta.hemi == 'lh' else 1
        dentro = np.flatnonzero(np.isin(vertices[lado], etiqueta.vertices)) + lado * n_izquierdo
        for i in dentro:                     # uno a uno: numpy repartiría la tupla entre los puntos
            region_de[i] = (nombre, lobulo, hemisferio)
        filas += [{'lóbulo': lobulo, 'hemisferio': hemisferio, 'aumento (dB)': v} for v in cociente_db[dentro]]

    puntos = pd.DataFrame(filas)
    lobulos = (puntos.groupby(['lóbulo', 'hemisferio'])['aumento (dB)']
               .agg(**{'aumento medio (dB)': 'mean', 'aumento máximo (dB)': 'max',
                       '% puntos activos': lambda v: 100 * (v >= umbral_db).mean()})
               .sort_values('aumento medio (dB)', ascending=False).round(1))

    # Punto que más crece, de entre los que tienen región en el atlas
    con_region = np.array([r is not None for r in region_de])
    i_max = np.flatnonzero(con_region)[cociente_db[con_region].argmax()]
    nombre, lobulo, hemisferio = region_de[i_max]
    maximo = {'región': nombre, 'lóbulo': lobulo, 'hemisferio': hemisferio,
              'aumento (dB)': float(cociente_db[i_max])}
    return maximo, lobulos
