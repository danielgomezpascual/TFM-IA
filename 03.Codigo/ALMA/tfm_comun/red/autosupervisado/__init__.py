"""Autosupervisado (F03.07): enseñar a la red "cómo es un EEG" con señal SIN etiquetas antes de enseñarle a predecir.

Imagen: fase_03/img/preentreno_autosupervisado.png (los dos pasos: preentrenar y ajustar)

    corpus_construccion.py -> qué EDF de 🔵 train entran y cómo se convierten en el corpus (log-FFT por segundo).
    corpus.py              -> Corpus: el corpus cargado en memoria y dónde empiezan sus secuencias.
    estadisticos_paciente.py -> mediana y escala de cada paciente, para normalizarlo frente a sí mismo.
    lotes_corpus.py        -> LotesCorpus: lotes de (60 s de contexto, 12 s de futuro), normalizados por paciente.
    perdida_futuro.py      -> el error al predecir el futuro (MAE con máscara) y el "suelo" a batir.
    preentrenar.py         -> PREENTRENO (ajustes) y preentrenar (el bucle de épocas, con puntos de control).
    transferencia_pesos.py -> copiar lo aprendido a la red de predicción y, si se quiere, congelarlo.

La red del preentrenamiento (RedPreentreno) está en red/redes/preentreno.py.
"""



