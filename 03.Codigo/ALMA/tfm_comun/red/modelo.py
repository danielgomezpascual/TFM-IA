"""
Modelo: el catálogo de redes neuronales de predicción (⏳ preictal frente a 💤 interictal) y cómo se crea una.

Motivo para cambiar: que se añada o se quite un tipo de red (cada red está en su propio fichero, en red/redes/).

Las redes
=========
    'cnn_gru'       RedCNNGRU       (red/redes/cnn_gru.py, F03.04)       CNN por ventana de 4 s + GRU sobre las ventanas
    'gru_segundos'  RedGRUSegundos  (red/redes/gru_segundos.py, F03.06)  GRU segundo a segundo, al estilo de Tang (sin grafo)
    'dcrnn'         RedDCRNN        (red/redes/dcrnn.py, F03.08)         la red de grafos de Tang (DCGRU)

Las arquitecturas (diccionarios con los tamaños de cada red) están junto a su red: ARQUITECTURA en cnn_gru.py,
ARQUITECTURA_SEGUNDOS y ARQUITECTURA_BANDAS en gru_segundos.py, ARQUITECTURA_DCRNN en dcrnn.py.
La red del preentrenamiento (RedPreentreno, F03.07) está en red/redes/preentreno.py: no predice crisis, así que no
está en el catálogo.

Para añadir una red nueva: un fichero nuevo en red/redes/ con la clase y su arquitectura, y una línea en REDES.
"""

from tfm_comun.red.redes.cnn_gru import ARQUITECTURA, RedCNNGRU
from tfm_comun.red.redes.dcrnn import RedDCRNN
from tfm_comun.red.redes.gru_segundos import RedGRUSegundos

REDES = {'cnn_gru': RedCNNGRU, 'gru_segundos': RedGRUSegundos, 'dcrnn': RedDCRNN}


def crear_red(n_canales, n_frecuencias, arquitectura=None):
    """Una red nueva con la arquitectura dada (por defecto, ARQUITECTURA: la RedCNNGRU de F03.04, red/redes/cnn_gru.py).

    La clave 'tipo' elige la red: 'cnn_gru' (por defecto, si no está), 'gru_segundos' (RedGRUSegundos, F03.06) o 'dcrnn'
    (RedDCRNN, F03.08). Cada una en su fichero de red/redes/.
    n_frecuencias: números por canal de cada ventana (40 con el espectro relativo; 160 = 4 s x 40 Hz con la log-FFT por segundo).
    """
    arquitectura = dict(arquitectura or ARQUITECTURA)
    tipo = arquitectura.pop('tipo', 'cnn_gru')
    return REDES[tipo](n_canales, n_frecuencias, **arquitectura)


def contar_parametros(modelo):
    """Parámetros (números que se aprenden) por pieza de la red y en total: {'codificador': ..., 'total': ...}."""
    partes = {nombre: sum(p.numel() for p in hijo.parameters()) for nombre, hijo in modelo.named_children()}
    partes['total'] = sum(partes.values())
    return partes
