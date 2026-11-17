"""
Redes: cada tipo de red neuronal en su propio fichero (un solo motivo para cambiar: su arquitectura).

    cnn_gru.py       -> RedCNNGRU (y su CodificadorVentana): CNN por ventana de 4 s + GRU sobre las ventanas (F03.04).
    gru_segundos.py  -> RedGRUSegundos: GRU segundo a segundo, al estilo de Tang sin el grafo (F03.06).
    dcrnn.py         -> RedDCRNN (y sus piezas ConvolucionDifusion y CeldaDCGRU): la red de grafos de Tang (F03.08).
    preentreno.py    -> RedPreentreno: RedGRUSegundos + decodificador que predice el futuro (F03.07).

Cada fichero lleva también su arquitectura por defecto (diccionario ARQUITECTURA...).
El catálogo de redes y crear_red están en red/modelo.py.
"""
