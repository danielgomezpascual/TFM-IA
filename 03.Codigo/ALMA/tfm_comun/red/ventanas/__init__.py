"""Ventanas: cómo le llegan a PyTorch las ventanas sueltas de 4 s en bruto (F03.01).

Imagen: fase_03/img/del_indice_al_lote.png (del índice de ventanas al lote que entra en la red)

Las tres piezas de PyTorch, cada una en su fichero (un solo motivo para cambiar cada una):
    dataset_ventanas.py -> VentanasEEG: el Dataset. Sabe cuántas ventanas hay y devuelve UNA (22 x 1.024) con su etiqueta.
    preparar_lote.py    -> PrepararLote: junta las ventanas de un lote, las normaliza y (al entrenar) las refleja.
    cargador.py         -> crear_cargador: el DataLoader, que pide las ventanas al Dataset y se las pasa a PrepararLote.

Por qué se normaliza al hacer el lote y no en el paquete: así es exactamente lo que hará la demo (normalizar la
ventana que acaba de llegar) y se puede cambiar la normalización sin rehacer los paquetes.

Las redes de F03.04 en adelante NO usan estas piezas: trabajan con el espectro de cada ventana (red/lotes/).
"""



