"""
Indicadores de calidad de un trozo de señal: amplitud, canales planos y canales que faltan.

Motivo para cambiar: que cambie cómo se mide la calidad de la señal.
Aquí NO se descarta nada: solo se mide. Qué tramos o ventanas se descartan se decide después
(03_ventanas), con estos indicadores guardados en senales.csv.
"""

import numpy as np

UMBRAL_PLANO_UV = 0.5   # µV de desviación típica: por debajo, el canal está plano (electrodo suelto o apagado)


def indicadores_calidad(senal, canales_validos):
    """Mide la calidad de un trozo de señal preprocesada (no descarta nada, solo mide).

    Qué hace:
        Mira solo los canales válidos (los que no se rellenaron con 0) y calcula:
        - amplitud_max_uv: el mayor valor absoluto. Muy alto (miles de µV) = probable artefacto.
        - desviacion_mediana_uv: la desviación típica mediana de los canales, el nivel "típico" de la señal.
        - canales_planos: canales con desviación < UMBRAL_PLANO_UV (0,5 µV): electrodo suelto o apagado.
        - canales_validos: cuántos canales se pudieron construir (22, o 20 sin A1/A2).

    Parámetros:
        senal: matriz (22 x muestras) en µV.
        canales_validos: 22 booleanos (de tcp_desde_referencial).

    Devuelve:
        Diccionario con esos 4 indicadores (se guardan como columnas en senales.csv).

    Imagen: tfm_comun/imagenes/calidad_indicadores.png
    """
    validos = np.asarray(canales_validos)
    senal_valida = np.asarray(senal)[validos]
    desviacion = senal_valida.std(axis=-1)
    return {
        'amplitud_max_uv': float(np.abs(senal_valida).max()),
        'desviacion_mediana_uv': float(np.median(desviacion)),
        'canales_planos': int((desviacion < UMBRAL_PLANO_UV).sum()),
        'canales_validos': int(validos.sum()),
    }
