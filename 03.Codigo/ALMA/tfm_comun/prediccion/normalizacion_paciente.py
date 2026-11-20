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

⚠️ Pega de normalizar_con_calibracion() (vista en F03.03, 2026-11-18): la referencia de cada ventana DEPENDE DE SU
ETIQUETA. Las 💤 se comparan con la otra mitad de las 💤 y las ⏳ con todas las 💤. Si el paciente cambia entre
grabaciones, las 💤 quedan más lejos del cero que las ⏳ solo por eso, y un modelo puede aprenderlo sin saber nada
de crisis. Además, una app no sabe la etiqueta al normalizar. Se deja para comparar (y la usa la red por ahora).

Versión limpia: reserva_calibracion() + normalizar_con_reserva(). Por paciente se RESERVAN sus primeras 💤 en el
tiempo (los "minutos tranquilos" de la app); su mediana es la referencia de TODAS sus demás ventanas, ⏳ y 💤 por
igual; y las reservadas no se usan ni para entrenar ni para medir.

Para la red (F03.04 y red/preparacion.py), 2026-11-19
=====================================================
reserva_de_filas() hace la reserva EXACTAMENTE como F03.03 §6 (mismo orden en el tiempo, mismos segundos), y
calibracion_para_red() da la calibración de cada fila con los MISMOS nombres y reglas que F03.03:
    'ninguna'              -> sin normalizar;
    'por_paciente'         -> la mediana de TODAS las ventanas del paciente (como normalizar_por_paciente);
    'por_paciente_reserva' -> la mediana de sus ventanas reservadas (como normalizar_con_reserva).
    'por_sesion_reserva'   -> lo mismo, pero por SESIÓN (F03.03b): cada grabación se calibra con sus primeras 💤,
                              como haría una app al empezar a grabar. Quita las diferencias de nivel entre días.
La calibración antigua con fuga ('por_paciente_calibrado') ya no se acepta en la red.
ventanas_del_estudio() reúne la reserva y las ventanas que se usan según la decisión (F03.03 o F03.03b).
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


def reserva_calibracion(pacientes, es_tranquila, orden, n_reserva, minimo):
    """Qué ventanas se reservan para calibrar a cada paciente (y qué pacientes se pueden calibrar).

    Imita una app: al empezar, graba unos minutos tranquilos del paciente para saber cuál es su "normal".
    Por paciente se reservan sus primeras 💤 en el tiempo: min(n_reserva, la mitad de sus 💤). Así siempre le
    quedan 💤 para medir. Si con eso no llega a `minimo` ventanas, el paciente no se puede calibrar.

    Parámetros:
        pacientes: paciente de cada ventana.  es_tranquila: True en las 💤 interictales.
        orden: posición de cada ventana en el tiempo (menor = antes), p. ej. de np.lexsort.
        n_reserva: ventanas que se reservan por paciente (p. ej. 30 = 2 min de ventanas de 4 s).
        minimo: ventanas de reserva por debajo de las cuales el paciente no se calibra.

    Devuelve:
        (reservada, calibrable): arrays True/False del largo de pacientes. reservada = ventana de calibración
        (no se entrena ni se mide con ella); calibrable = ventana de un paciente con calibración. Las ventanas
        con calibrable False tampoco se usan (no hay un "normal" honesto con el que compararlas).
    """
    pacientes = np.asarray(pacientes)
    es_tranquila = np.asarray(es_tranquila, dtype=bool)
    orden = np.asarray(orden)
    reservada = np.zeros(len(pacientes), dtype=bool)
    calibrable = np.zeros(len(pacientes), dtype=bool)
    for _, posiciones in pd.Series(pacientes).groupby(pacientes).indices.items():
        tranquilas = posiciones[es_tranquila[posiciones]]
        tranquilas = tranquilas[np.argsort(orden[tranquilas], kind='stable')]        # de la más antigua a la más nueva
        n = min(n_reserva, len(tranquilas) // 2)
        if n < minimo:
            continue
        reservada[tranquilas[:n]] = True
        calibrable[posiciones] = True
    return reservada, calibrable


def normalizar_con_reserva(X, pacientes, reservada):
    """Cada paciente frente a su calibración reservada: la MISMA referencia para todas sus ventanas, ⏳ y 💤.

    Para cada paciente: (valor - mediana de sus ventanas reservadas) / (rango intercuartílico de ellas / 1,349).
    La referencia no depende de la etiqueta de la ventana que se normaliza (a diferencia de normalizar_con_calibracion).

    Parámetros:
        X: (ventanas, características).  pacientes: paciente de cada ventana.
        reservada: True en las ventanas de calibración (reserva_calibracion()).

    Devuelve:
        Array float32 con la misma forma. Las filas de pacientes sin ventanas reservadas quedan NaN (no se usan).
    """
    X = np.asarray(X, dtype=np.float32)
    reservada = np.asarray(reservada, dtype=bool)
    salida = np.full_like(X, np.nan)
    for _, posiciones in pd.Series(np.asarray(pacientes)).groupby(np.asarray(pacientes)).indices.items():
        de_reserva = posiciones[reservada[posiciones]]
        if len(de_reserva) == 0:
            continue
        mediana, escala = _estadisticos(X[de_reserva])
        salida[posiciones] = (X[posiciones] - mediana) / escala
    return salida


NORMALIZACIONES_RED = ('ninguna', 'por_paciente', 'por_paciente_reserva', 'por_sesion_reserva')   # las de F03.03 (+ la de F03.03b)


def reserva_de_filas(filas, fs, ventana_s, reserva_s, minimo_s, por='paciente'):
    """La reserva de calibración de F03.03 (§6), calculada igual a partir de la tabla de filas.

    Ordena las ventanas en el tiempo (por fichero y segundo dentro del fichero) y llama a reserva_calibracion() con
    las 💤 como ventanas tranquilas. Así F03.04 y red/preparacion.py apartan EXACTAMENTE las mismas ventanas que F03.03.

    Parámetros:
        filas: tabla de ventanas (las de contexto completo de F03.03), con ruta, muestra_inicio, file, patient y etiqueta.
        fs: muestras por segundo.  ventana_s: segundos por ventana.
        reserva_s, minimo_s: los de F03.03 (config_03['reserva_calibracion']['segundos'] y ['minimo_s']).
        por: 'paciente' (F03.03) o 'sesion' (F03.03b: se reserva al principio de cada grabación).

    Devuelve:
        (reservada, calibrable, usar): arrays True/False del largo de filas. usar = calibrable & ~reservada: las
        ventanas con las que F03.03 entrena y mide (en TODAS sus normalizaciones).
    """
    from tfm_comun.prediccion.contextos.disponibilidad import inicio_tramo_s      # aquí dentro: evita importaciones en círculo
    from tfm_comun.prediccion.contextos.duracion import ventanas_por_contexto

    segundo_en_fichero = inicio_tramo_s(filas.ruta) + filas.muestra_inicio.to_numpy() / fs
    orden_tiempo = np.lexsort((segundo_en_fichero, filas.file.astype(str).to_numpy()))
    posicion_tiempo = np.empty(len(filas), dtype=np.int64)
    posicion_tiempo[orden_tiempo] = np.arange(len(filas))
    assert por in ('paciente', 'sesion'), f'por debe ser paciente o sesion, no {por}'
    grupos = filas.patient.to_numpy() if por == 'paciente' else sesiones_de(filas)
    reservada, calibrable = reserva_calibracion(grupos, filas.etiqueta.to_numpy() == 0, posicion_tiempo,
                                                n_reserva=ventanas_por_contexto(reserva_s, ventana_s),
                                                minimo=ventanas_por_contexto(minimo_s, ventana_s))
    return reservada, calibrable, calibrable & ~reservada


def calibracion_para_red(X, pacientes, normalizacion, reservada=None):
    """La calibración de cada fila para la red, con las reglas de F03.03 (sin aplicarla).

    Parámetros:
        X: (ventanas, características): la característica de la ventana ACTUAL de cada fila (p. ej. su espectro).
        pacientes: paciente de cada fila; con 'por_sesion_reserva', la SESIÓN de cada fila (ventanas_del_estudio da los grupos).
        normalizacion: una de NORMALIZACIONES_RED.
        reservada: True en las ventanas reservadas para calibrar (reserva_de_filas); hace falta con 'por_paciente_reserva'.

    Devuelve:
        (grupo, medianas, escalas), como calibracion_por_paciente(): normalizar la fila i es
        (X[i] - medianas[grupo[i]]) / escalas[grupo[i]]. Con 'ninguna', una sola calibración (mediana 0, escala 1).
        Con 'por_paciente_reserva', las filas de pacientes sin reserva quedan con grupo -1: no se pueden usar
        (son las mismas que F03.03 deja fuera; hay que filtrarlas con `usar` de reserva_de_filas).
    """
    X = np.asarray(X, dtype=np.float32)
    pacientes = np.asarray(pacientes)
    if normalizacion == 'por_paciente_calibrado':
        raise ValueError("'por_paciente_calibrado' es la calibración antigua con fuga por la etiqueta (la referencia de cada ventana "
                         "depende de si es ⏳ o 💤). Usa 'por_paciente' o 'por_paciente_reserva', como F03.03.")
    assert normalizacion in NORMALIZACIONES_RED, f'Normalización desconocida: {normalizacion}. Opciones: {NORMALIZACIONES_RED}'
    if normalizacion == 'ninguna':
        return (np.zeros(len(X), dtype=np.int64), np.zeros((1, X.shape[1]), dtype=np.float32),
                np.ones((1, X.shape[1]), dtype=np.float32))
    if normalizacion == 'por_paciente':
        referencia = np.ones(len(X), dtype=bool)
    else:
        assert reservada is not None, f"Con {normalizacion!r} hay que dar `reservada` (reserva_de_filas o ventanas_del_estudio)."
        referencia = np.asarray(reservada, dtype=bool)
    grupo = np.full(len(X), -1, dtype=np.int64)
    medianas, escalas = [], []
    for _, posiciones in pd.Series(pacientes).groupby(pacientes).indices.items():
        de_referencia = posiciones[referencia[posiciones]]
        if len(de_referencia) == 0:
            continue
        mediana, escala = _estadisticos(X[de_referencia])
        grupo[posiciones] = len(medianas)
        medianas.append(mediana)
        escalas.append(escala)
    if not medianas:
        raise ValueError(f'Ningún paciente tiene ventanas de referencia para {normalizacion}.')
    return grupo, np.asarray(medianas, dtype=np.float32), np.asarray(escalas, dtype=np.float32)


def sesiones_de(filas):
    """La sesión de cada fila: 'aaaaaajy_s003_t004' -> 'aaaaaajy_s003' (array de str)."""
    return filas.file.astype(str).str.rsplit('_', n=1).str[0].to_numpy()


def ventanas_del_estudio(filas, fs, ventana_s, reserva, normalizacion, reserva_por_sesion=False):
    """Las ventanas que se usan y la reserva con la que se calibra, según la decisión para la red.

    Parámetros:
        filas: las filas con contexto completo (las de F03.03).  fs, ventana_s: de F03.01.
        reserva: config_03['reserva_calibracion'] (segundos y minimo_s).
        normalizacion: una de NORMALIZACIONES_RED.
        reserva_por_sesion: True si la decisión viene de F03.03b, que aparta TAMBIÉN la reserva de cada sesión
            (así todas sus variantes se miden con las mismas ventanas).

    Devuelve:
        (reservada, usar, grupos): reservada = ventanas de calibración de la normalización elegida; usar = las que
        se entrenan y miden; grupos = paciente o sesión de cada fila (con qué se agrupa la calibración).
    """
    res_p, _, usar = reserva_de_filas(filas, fs, ventana_s, reserva['segundos'], reserva['minimo_s'], por='paciente')
    reservada, grupos = res_p, filas.patient.to_numpy()
    if reserva_por_sesion or normalizacion == 'por_sesion_reserva':
        res_s, _, usar_s = reserva_de_filas(filas, fs, ventana_s, reserva['segundos'], reserva['minimo_s'], por='sesion')
        usar = usar & usar_s
        if normalizacion == 'por_sesion_reserva':
            reservada, grupos = res_s, sesiones_de(filas)
    return reservada, usar, grupos
