"""
Normalización por paciente: expresar las características de cada paciente respecto a lo normal EN ÉL.

Motivo para cambiar: que cambie cómo se compara a cada paciente consigo mismo.

Imagen: fase_03/img/normalizacion_paciente.png (dos pacientes con niveles distintos, antes y después)

Por qué
=======
F03.02 vio que, dentro de cada paciente, el modelo ordena las ventanas algo mejor que mezclando pacientes
(AUC por paciente > AUC global): cada persona tiene su "nivel" propio (su reparto de energía entre bandas,
su equipo, su medicación). Un mismo número puede ser alto en uno y normal en otro. Si a cada paciente se le
resta su valor típico, el modelo ve "cuánto se aparta esta ventana de lo normal en esta persona".

Cómo (sin hacer trampa)
=======================
Para cada paciente y cada característica: (valor - mediana del paciente) / (rango intercuartílico / 1,349).
La mediana y el rango se calculan con TODAS las ventanas de ese paciente del grupo, SIN mirar su etiqueta
(como haría una app con unos minutos de grabación de calibración). Limitación a contar en la memoria: las
ventanas ⏳ de ese paciente también entran en su mediana, y en 🟡 dev la proporción de ⏳ es distinta que en
🔵 train (equilibrado mitad y mitad), así que la "calibración" no es idéntica en los dos.

Para comprobar que la mejora no viene de ahí, normalizar_con_calibracion() calibra a cada paciente SOLO con sus
💤 interictales (como una app que graba unos minutos tranquilos), sin que ninguna ventana se calibre consigo misma.
"""

import warnings

import numpy as np
import pandas as pd

ESCALA_IQR = 1.349        # rango intercuartílico de una normal estándar: así la escala se parece a la de un z-score
EPSILON = 1e-6


def normalizar_por_paciente(X, pacientes):
    """Cada característica, centrada y escalada con la mediana y el rango intercuartílico de su paciente.

    Parámetros:
        X: (ventanas, características).  pacientes: paciente de cada ventana.

    Devuelve:
        Array float32 con la misma forma. Los NaN se quedan NaN (y no cuentan para la mediana).
    """
    X = np.asarray(X, dtype=np.float32)
    salida = np.empty_like(X)
    for _, posiciones in pd.Series(np.asarray(pacientes)).groupby(np.asarray(pacientes)).indices.items():
        bloque = X[posiciones]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)   # columnas de solo NaN (canales sin señal)
            mediana = np.nanmedian(bloque, axis=0)
            q1, q3 = np.nanpercentile(bloque, [25, 75], axis=0)
        salida[posiciones] = (bloque - mediana) / ((q3 - q1) / ESCALA_IQR + EPSILON)
    return salida


def _estadisticos(bloque):
    """Mediana y escala (rango intercuartílico / 1,349) de cada columna, ignorando los NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mediana = np.nanmedian(bloque, axis=0)
        q1, q3 = np.nanpercentile(bloque, [25, 75], axis=0)
    return mediana, (q3 - q1) / ESCALA_IQR + EPSILON


def calibracion_por_paciente(X, pacientes, referencia):
    """La "calibración tranquila" de cada paciente, sin aplicarla: con qué mediana y escala se normaliza cada fila.

    Las reglas son las de normalizar_con_calibracion() (abajo), que la usa:
        - la calibración de cada paciente sale SOLO de sus ventanas de referencia (sus 💤 interictales);
        - ninguna ventana se calibra consigo misma: las de referencia se parten en dos mitades seguidas (en el orden
          de las filas) y cada mitad se normaliza con la mediana de la OTRA;
        - las demás ventanas del paciente (sus ⏳) se normalizan con todas sus ventanas de referencia;
        - un paciente con menos de 2 ventanas de referencia se calibra con todas las suyas.

    Por qué separarla: la red de F03.04 normaliza secuencias de 15 ventanas en cada lote. Con esto basta guardar
    unas pocas medianas por paciente (y no una copia normalizada de todo).

    Parámetros:
        X: (ventanas, características).  pacientes: paciente de cada ventana.
        referencia: True en las ventanas que pueden servir de calibración (las 💤 interictales).

    Devuelve:
        (grupo, medianas, escalas): grupo = int por fila (qué calibración le toca); medianas y escalas = arrays
        float32 (grupos, características). Normalizar la fila i es (X[i] - medianas[grupo[i]]) / escalas[grupo[i]].
    """
    X = np.asarray(X, dtype=np.float32)
    referencia = np.asarray(referencia, dtype=bool)
    grupo = np.full(len(X), -1, dtype=np.int64)
    medianas, escalas = [], []

    def nuevo(con, a):
        mediana, escala = _estadisticos(X[con])
        grupo[a] = len(medianas)
        medianas.append(mediana)
        escalas.append(escala)

    for _, posiciones in pd.Series(np.asarray(pacientes)).groupby(np.asarray(pacientes)).indices.items():
        de_referencia = posiciones[referencia[posiciones]]
        resto = posiciones[~referencia[posiciones]]
        if len(de_referencia) < 2:
            nuevo(posiciones, posiciones)
            continue
        mitad = len(de_referencia) // 2
        primera, segunda = de_referencia[:mitad], de_referencia[mitad:]
        nuevo(segunda, primera)
        nuevo(primera, segunda)
        if len(resto):
            nuevo(de_referencia, resto)
    return grupo, np.asarray(medianas, dtype=np.float32), np.asarray(escalas, dtype=np.float32)


def normalizar_con_calibracion(X, pacientes, referencia):
    """Como normalizar_por_paciente(), pero la "calibración" de cada paciente sale SOLO de sus ventanas de referencia.

    Es la versión más parecida a una app real: la mediana y la escala de cada paciente se calculan con un registro
    tranquilo suyo (aquí, sus 💤 interictales), sin que sus ⏳ preictales influyan. Así se descarta que la mejora
    venga de que la proporción de ⏳ de cada paciente mueva su mediana.

    Ninguna ventana se calibra consigo misma: las ventanas de referencia de cada paciente se parten en dos mitades
    seguidas en el tiempo (en el orden de las filas), y cada mitad se normaliza con la mediana de la OTRA mitad.
    Las demás ventanas del paciente (sus ⏳) se normalizan con todas sus ventanas de referencia.
    Las reglas están en calibracion_por_paciente(); aquí solo se aplican.

    Parámetros:
        X: (ventanas, características).  pacientes: paciente de cada ventana.
        referencia: True en las ventanas que pueden servir de calibración (las 💤 interictales).

    Devuelve:
        Array float32 con la misma forma. Un paciente con menos de 2 ventanas de referencia se calibra con todas las suyas.
    """
    X = np.asarray(X, dtype=np.float32)
    grupo, medianas, escalas = calibracion_por_paciente(X, pacientes, referencia)
    return ((X - medianas[grupo]) / escalas[grupo]).astype(np.float32)
