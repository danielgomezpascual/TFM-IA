"""
Características de una ventana: de 22 x 1.024 números a 110 números que un modelo clásico entiende.

Motivo para cambiar: que cambie qué se mide de cada ventana (bandas, forma de calcular el espectro...).
Trabaja con arrays de numpy (sin MNE): sirve igual para el TFM y para la demo.

Imagen: fase_03/img/de_ventana_a_numeros.png (de la ventana al espectro, a las bandas y a los 110 números)

Qué se mide: potencia RELATIVA por banda y canal
================================================
1. Espectro de cada canal con el método de Welch: se parte la ventana de 4 s en trozos de 1 s que se
   solapan a la mitad, se calcula la FFT de cada trozo y se promedian. Sale cuánta energía hay en cada
   frecuencia (resolución de 1 Hz), más estable que con una sola FFT.
2. Energía de cada banda clásica (senal/bandas.py: delta, theta, alfa, beta, gamma): suma del espectro
   dentro de la banda.
3. Potencia RELATIVA: la energía de cada banda dividida por la de todas las bandas juntas (suman 1).
4. Logaritmo en base 10: las potencias van de 0,001 a 0,9; con el logaritmo quedan en una escala parecida.

Resultado: 22 canales x 5 bandas = 110 números por ventana, con nombres como 'FP1-F7 · delta'.

Por qué RELATIVA y no la energía en µV²
=======================================
F02.05 vio que el nivel de señal (lo "fuerte" que se mide un canal) separa un poco las clases (H3) y es una
pista tramposa: depende del equipo y del paciente. La potencia relativa no cambia si la señal se multiplica
por cualquier número (como la normalización por ventana y canal de F03.01), así que el modelo solo ve
CÓMO se reparte la energía entre bandas, no cuánta hay.

Canales sin señal (A1-T3 y T4-A2 en ficheros sin electrodos A1/A2, que F02.02 rellena con ceros): su energía
total es 0 y la potencia relativa no existe. Quedan como NaN ("no es un número"); los modelos de
prediccion/linea_base.py saben tratarlos.

Nota: la señal viene filtrada a 0,5-40 Hz (senal/filtrado.py), así que la banda gamma (30-45 Hz) solo tiene
de verdad la parte de 30 a 40 Hz.
"""

import numpy as np
from scipy.signal import welch

from tfm_comun.senal.bandas import BANDAS_HZ

SEGMENTO_S = 1.0          # trozos de 1 s para el espectro de Welch (resolución de 1 Hz)
POTENCIA_MINIMA = 1e-6    # suelo de la potencia relativa antes del logaritmo (log10(0) no existe)
ENERGIA_MINIMA = 1e-12    # por debajo de esto, el canal se considera sin señal (relleno con ceros)


def nombres_caracteristicas(canales, bandas=BANDAS_HZ):
    """Nombre de cada característica, en el mismo orden que calcular_caracteristicas(): 'FP1-F7 · delta'..."""
    return [f'{canal} · {banda}' for canal in canales for banda in bandas]


def potencia_por_banda(ventanas, fs, bandas=BANDAS_HZ):
    """Energía de cada banda en cada canal (espectro de Welch).

    Parámetros:
        ventanas: array (ventanas, canales, muestras), en µV.  fs: muestras por segundo (256).
        bandas: {nombre: (Hz inferior, Hz superior)}.

    Devuelve:
        Array float32 (ventanas, canales, bandas) con la energía (µV²) de cada banda.
    """
    ventanas = np.asarray(ventanas, dtype=np.float32)
    nperseg = int(round(SEGMENTO_S * fs))
    frecuencias, espectro = welch(ventanas, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    paso = frecuencias[1] - frecuencias[0]
    energia = [espectro[..., (frecuencias >= baja) & (frecuencias < alta)].sum(axis=-1) * paso
               for baja, alta in bandas.values()]
    return np.stack(energia, axis=-1).astype(np.float32)


def calcular_caracteristicas(ventanas, fs, bandas=BANDAS_HZ):
    """Las 110 características de cada ventana: log10 de la potencia relativa por canal y banda.

    Qué hace:
        1. Energía por banda y canal (potencia_por_banda).
        2. La divide por la energía total del canal (suma de las bandas): potencia relativa, suma 1.
        3. log10, con un suelo de POTENCIA_MINIMA. Los canales sin señal quedan como NaN.

    Parámetros:
        ventanas: array (ventanas, canales, muestras), en µV (no hace falta normalizar: la relativa no
            depende de la escala).  fs: muestras por segundo.

    Devuelve:
        Array float32 (ventanas, canales x bandas), en el orden de nombres_caracteristicas().

    Ejemplo: una ventana con mucha delta y poca beta en FP1-F7 da 'FP1-F7 · delta' ≈ -0,2 (63 % de la
    energía) y 'FP1-F7 · beta' ≈ -1,5 (3 %).
    """
    energia = potencia_por_banda(ventanas, fs, bandas)
    total = energia.sum(axis=-1, keepdims=True)
    relativa = np.where(total > ENERGIA_MINIMA, energia / np.maximum(total, ENERGIA_MINIMA), np.nan)
    caracteristicas = np.log10(np.maximum(relativa, POTENCIA_MINIMA))     # NaN se queda NaN
    return caracteristicas.reshape(len(caracteristicas), -1).astype(np.float32)


def caracteristicas_por_trozos(matriz, fs, posiciones=None, trozo=2048, progreso=None):
    """calcular_caracteristicas() sobre un paquete entero (mmap), a trozos para no llenar la memoria.

    Parámetros:
        matriz: array (ventanas, canales, muestras), p. ej. el paquete abierto con mmap (red/paquetes.py).
        posiciones: filas que se quieren (por defecto, todas).  trozo: ventanas por trozo.
        progreso: función tipo tqdm para envolver el bucle (opcional).

    Devuelve:
        Array float32 (len(posiciones), características).
    """
    posiciones = np.arange(len(matriz)) if posiciones is None else np.asarray(posiciones)
    inicios = range(0, len(posiciones), trozo)
    recorrido = progreso(inicios, total=len(inicios)) if progreso else inicios
    partes = [calcular_caracteristicas(np.asarray(matriz[posiciones[a:a + trozo]], dtype=np.float32), fs)
              for a in recorrido]
    return np.concatenate(partes) if partes else np.empty((0, 0), dtype=np.float32)
