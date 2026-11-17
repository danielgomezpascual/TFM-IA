"""Cálculo del contexto: lo que se mide de cada ventana que hace falta, leyéndola de la señal de F02.02.

Motivo para cambiar: que cambie cómo se lee la señal o se reparte el cálculo.

Se sacan siempre de la señal de F02.02 (el tramo de la ventana, que está completo), y no de los paquetes:
F02.04 sorteó las 💤 interictales para equilibrar, así que en los paquetes a muchas les faltan las vecinas.
Así el contexto se construye IGUAL para ⏳ y para 💤 (si no, sería una pista tramposa).
La función que mide cada ventana se recibe como parámetro (`calcular`): las 110 características de XGBoost o
cualquiera de las entradas de la red (red/entradas/). Este fichero no necesita saber cuál.
"""

import numpy as np

from tfm_comun.prediccion.caracteristicas import calcular_caracteristicas
from tfm_comun.prediccion.ventanas import ventanas_de_senal
from tfm_comun.senal.senales import cargar_tramo


def caracteristicas_de_inicios(tabla, rutas_senales, fs, tamano, progreso=None, trozo=2048, calcular=calcular_caracteristicas):
    """Lo que se mide de cada ventana de la tabla (por defecto, las 110 características), leyéndola de la señal de F02.02.

    Parámetros:
        tabla: de inicios_necesarios() (columnas ruta e inicio).  rutas_senales: carpeta de la señal (RUTA_SENALES).
        fs: muestras por segundo.  tamano: muestras por ventana.  progreso: función tipo tqdm (opcional).
        calcular: función (ventanas, fs) -> (ventanas, números). Por defecto calcular_caracteristicas (F03.02/F03.03);
            la red de F03.04 pasa red/entradas/espectro_relativo.py (el espectro entero, no 5 bandas).

    Devuelve:
        Array (filas de la tabla, números por ventana), en el mismo orden que la tabla, del tipo que dé `calcular`
        (float32; float16 con la log-FFT por segundo de F03.06, que tiene 4 veces más números por ventana).
    """
    salida = None
    grupos = tabla.groupby('ruta', sort=False).indices
    recorrido = progreso(grupos.items(), total=len(grupos)) if progreso else grupos.items()
    for ruta, posiciones in recorrido:
        senal = cargar_tramo(rutas_senales / ruta)
        inicios = tabla.inicio.to_numpy()[posiciones]
        for a in range(0, len(posiciones), trozo):
            ventanas = np.asarray(ventanas_de_senal(senal, tamano, inicios[a:a + trozo]), dtype=np.float32)
            valores = calcular(ventanas, fs)
            if salida is None:
                salida = np.full((len(tabla), valores.shape[1]), np.nan, dtype=valores.dtype)   # float32, o float16 si calcular lo da así
            salida[posiciones[a:a + trozo]] = valores
        del senal
    return salida if salida is not None else np.empty((0, 0), dtype=np.float32)
