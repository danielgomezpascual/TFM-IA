"""
Normalización de la señal, ventana a ventana y canal a canal.

Motivo para cambiar: que cambie cómo se normaliza la señal antes de entrar al modelo.
Trabaja con arrays de numpy, sin MNE, para usarlo igual en el TFM y en la demo.

Por qué por ventana y no con medias de todo el dataset: cada paciente, equipo y electrodo
tiene una amplitud distinta. Normalizando cada ventana con sus propios valores, el modelo
no depende de eso, y en la demo basta con la ventana que acaba de llegar (no hace falta
guardar estadísticas de entrenamiento).

La señal preprocesada (02_preprocesado) se guarda SIN normalizar, en µV: se normaliza al cortar
las ventanas (03_ventanas), para poder probar distintas normalizaciones sin repetir el preprocesado.

Imagen: tfm_comun/imagenes/normalizacion_zscore_robusta.png (las dos opciones frente a un artefacto)
"""

import numpy as np

EPSILON = 1e-6   # evita dividir por 0 en canales planos o que faltan (quedan a 0)


def zscore_por_canal(ventana):
    """Normaliza cada canal de una ventana con su media y su desviación típica (z-score).

    Qué hace:
        Para cada canal: (x - media) / desviación típica. Después, cada canal tiene media 0 y
        desviación 1, así que un canal de 200 µV y otro de 20 µV quedan en la misma escala.
        Pega: un solo pico enorme (artefacto) infla la desviación y "aplasta" el resto del canal.

    Parámetros:
        ventana: array (..., canales, muestras). Puede llevar dimensiones delante (p. ej. un lote de ventanas).

    Devuelve:
        Array float32 con la misma forma, en "desviaciones típicas". Los canales a 0 se quedan a 0 (EPSILON).

    Imagen: tfm_comun/imagenes/normalizacion_zscore_robusta.png
    """
    ventana = np.asarray(ventana, dtype=np.float32)
    media = ventana.mean(axis=-1, keepdims=True)
    desviacion = ventana.std(axis=-1, keepdims=True)
    return (ventana - media) / (desviacion + EPSILON)


def robusta_por_canal(ventana):
    """Normaliza cada canal con su mediana y su rango intercuartílico (versión robusta del z-score).

    Qué hace:
        Para cada canal: (x - mediana) / (IQR / 1,349). La mediana y el IQR (percentil 75 - percentil 25)
        apenas cambian por unos pocos valores extremos, así que un artefacto no aplasta el resto de la
        ventana. Se divide el IQR por 1,349 porque ese es el IQR de una normal estándar: así, con señal
        "normal", el resultado se parece al del z-score.

    Parámetros:
        ventana: array (..., canales, muestras).

    Devuelve:
        Array float32 con la misma forma.

    Imagen: tfm_comun/imagenes/normalizacion_zscore_robusta.png
    """
    ventana = np.asarray(ventana, dtype=np.float32)
    mediana = np.median(ventana, axis=-1, keepdims=True)
    q1, q3 = np.percentile(ventana, [25, 75], axis=-1, keepdims=True)
    return (ventana - mediana) / ((q3 - q1) / 1.349 + EPSILON)


NORMALIZACIONES = {'zscore': zscore_por_canal, 'robusta': robusta_por_canal}
