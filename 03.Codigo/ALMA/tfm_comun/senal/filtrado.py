"""
Filtrado de la señal: paso banda y notch de la red eléctrica.

Motivo para cambiar: que cambien las frecuencias de corte o el tipo de filtro.
Trabaja con arrays de numpy (canales x muestras), sin MNE, para poder usarlo
igual en el TFM y en la demo en tiempo real.

Qué filtro se usa y por qué
===========================

Son dos filtros encadenados.

Imagen: tfm_comun/imagenes/filtrado_explicacion.png (por qué Butterworth, el orden, el notch y causal frente
a fase cero, con señal real)

1. PASO BANDA Butterworth de orden 4, de 0,5 a 40 Hz
   - Paso banda: deja pasar una banda de frecuencias y quita lo que queda por debajo y por encima.
       · Por debajo de 0,5 Hz: derivas lentas (sudor, movimiento del electrodo, respiración).
       · Por encima de 40 Hz: actividad muscular (mandíbula, frente) y ruido de la red eléctrica.
       · Lo que importa en las crisis está en medio: delta, theta, alfa y beta (la ausencia, a ~3 Hz).
   - Butterworth: es el filtro MÁS PLANO en la banda que deja pasar (de 1 a 30 Hz no sube ni baja nada
     la señal). Otros (Chebyshev, elíptico) cortan con más pendiente, pero ONDULAN la banda que dejan pasar:
     unas frecuencias saldrían más fuertes que otras y se deformarían las ondas del EEG. Es el estándar en EEG.
   - Orden 4: lo empinada que es la "cuesta" del corte. No es un muro: a 40 Hz pasa el 71 % de la amplitud
     (−3 dB, la frecuencia de corte), a 50 Hz el 31 %, a 70 Hz el 4 % y a 100 Hz el 0,1 %.
     Más orden = cuesta más empinada, pero más "ecos" alrededor de las puntas y más retraso. 4 es el término
     medio habitual. (Al ser paso banda, el filtro real es de orden 8: 4 por abajo y 4 por arriba.)

2. NOTCH (filtro de muesca) a 60 Hz, con Q = 30
   - Notch: quita UNA frecuencia concreta y deja el resto casi intacto. Aquí, el zumbido de la red eléctrica,
     que en EE. UU. (donde se grabó TUSZ) va a 60 Hz. (En Europa sería 50 Hz.)
   - Q (factor de calidad): lo estrecha que es la muesca. Anchura = 60 / Q = 2 Hz (de 59 a 61 Hz).
     Q alto = muesca más estrecha (toca menos frecuencias vecinas, pero tarda más en "engancharse").
   - El paso banda ya baja los 60 Hz al ~12 %; el notch los quita del todo (pico de la gráfica del espectro).

Cómo se aplica
==============

- CAUSAL (sosfilt): solo usa el pasado de la señal, como pasará en la demo (la señal llega en directo y no
  se puede mirar el futuro). Si se entrena con el mismo filtro, el modelo ve en el TFM exactamente la misma
  señal que verá en la demo. Contrapartida: retrasa la señal, y no igual en todas las frecuencias:
  ~12 ms a 10-20 Hz, ~33 ms a 3 Hz (la ausencia) y ~0,25 s en las ondas muy lentas de 1 Hz. Las ondas
  se deforman un poco, pero igual en entrenamiento y en la demo, así que el modelo no se ve afectado.
  (Un filtro de fase cero, sosfiltfilt o el de MNE por defecto, no retrasa, pero usa unas décimas de segundo
  del futuro: no se podría reproducir en directo.)
- Formato 'sos' (second-order sections): la receta del filtro se guarda troceada en pequeños filtros de
  segundo orden encadenados. Si se calcula de una pieza (formato 'ba'), con frecuencias tan bajas como
  0,5 Hz los errores de redondeo pueden hacer que el filtro se vuelva inestable. 'sos' es la forma segura.
- Estado inicial: al arrancar, el filtro "cree" que la señal llevaba siempre en su primer valor. Así no
  aparece un salto artificial al principio (el filtro necesita "calentar").
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

# ── Paso banda ──────────────────────────────────────────────────────────────────────────────────────
PASO_BANDA_HZ = (0.5, 40)   # (corte inferior, corte superior) en Hz, decidido en el EDA de la Fase 1.
                            # En cada corte la amplitud baja al 71 % (−3 dB); fuera, cae poco a poco.
ORDEN_PASO_BANDA = 4        # Pendiente del corte: 2 = suave, 4 = habitual en EEG, 8 = casi un muro (más ecos).

# ── Notch de la red eléctrica ───────────────────────────────────────────────────────────────────────
NOTCH_HZ = 60               # Red eléctrica de EE. UU., donde se grabó TUSZ (en Europa sería 50 Hz).
CALIDAD_NOTCH = 30          # Q: anchura de la muesca = NOTCH_HZ / Q = 2 Hz. Más Q = muesca más estrecha.


def disenar_filtro(fs):
    """Receta (coeficientes) del filtro completo, paso banda + notch, para una señal muestreada a fs Hz.
    No filtra nada: devuelve la receta que luego aplican filtrar() o FiltroContinuo.

    - butter(...): diseña el Butterworth.
        · ORDEN_PASO_BANDA: pendiente del corte.
        · PASO_BANDA_HZ: frecuencias de corte (−3 dB).
        · btype='bandpass': deja pasar la banda. Otros: 'lowpass' (solo quita lo rápido),
          'highpass' (solo quita lo lento), 'bandstop' (quita una banda).
        · fs: frecuencia de muestreo; la necesita para saber qué son "40 Hz" en muestras.
        · output='sos': receta troceada en secciones de segundo orden (estable, ver arriba).
    - iirnotch(...): diseña el notch; devuelve la receta en formato 'ba' y tf2sos la pasa a 'sos'.
    - np.vstack: encadena las dos recetas en una sola (primero el paso banda, luego el notch).

    Parámetros:
        fs: frecuencia de muestreo de la señal que se va a filtrar (Hz). En el TFM, siempre 256.

    Devuelve:
        Array 'sos' (secciones x 6): la receta del filtro, lista para sosfilt.

    Imagen: tfm_comun/imagenes/filtrado_disenar_filtro.png (qué % de cada frecuencia deja pasar)
    """
    paso_banda = butter(ORDEN_PASO_BANDA, PASO_BANDA_HZ, btype='bandpass', fs=fs, output='sos')
    notch = tf2sos(*iirnotch(NOTCH_HZ, CALIDAD_NOTCH, fs=fs))
    return np.vstack([paso_banda, notch])


def _estado_inicial(sos, primera_muestra):
    """Estado inicial del filtro, para que no aparezca un salto artificial al arrancar.

    Qué hace:
        Un filtro tiene "memoria" (su estado). Si arranca con la memoria vacía (ceros) y la señal empieza
        lejos de 0, el filtro ve un escalón brusco y produce un pico falso durante el primer segundo.
        Aquí se calcula la memoria que tendría el filtro si la señal llevara siempre en su primer valor,
        así arranca suave. sosfilt_zi da ese estado para una entrada de valor 1; se escala por la primera
        muestra de cada canal.

    Parámetros:
        sos: receta del filtro (de disenar_filtro).
        primera_muestra: primer valor de cada canal (array de tamaño canales).

    Devuelve:
        Array (secciones x canales x 2) con el estado inicial, para pasarlo como zi= a sosfilt.

    Imagen: tfm_comun/imagenes/filtrado_estado_inicial.png
    """
    return sosfilt_zi(sos)[:, None, :] * np.asarray(primera_muestra)[None, :, None]


def filtrar(senal, fs):
    """Filtra una señal completa: paso banda 0,5-40 Hz + notch 60 Hz, de forma causal.

    Qué hace:
        Diseña el filtro para fs (disenar_filtro), le pone el estado inicial (_estado_inicial) y lo aplica
        con sosfilt a todos los canales a la vez. Quita la deriva lenta (< 0,5 Hz), el zumbido de la red
        (60 Hz) y lo rápido (> 40 Hz, sobre todo músculo). Es causal: solo usa el pasado de la señal,
        igual que se podrá hacer en directo en la demo.

    Parámetros:
        senal: matriz (canales x muestras).
        fs: frecuencia de muestreo (Hz).

    Devuelve:
        Matriz (canales x muestras) filtrada, en float64 y en la misma unidad que la entrada.

    Imagen: tfm_comun/imagenes/filtrado_filtrar.png
    """
    senal = np.asarray(senal, dtype=np.float64)
    sos = disenar_filtro(fs)
    filtrada, _ = sosfilt(sos, senal, axis=-1, zi=_estado_inicial(sos, senal[:, 0]))
    return filtrada


class FiltroContinuo:
    """El mismo filtro que filtrar(), pero para señal que llega a trozos (demo en tiempo real):
    guarda el estado entre un trozo y el siguiente, así que filtrar todo de golpe o a trozos da lo mismo.

    Uso:
        filtro = FiltroContinuo(fs=256)
        for bloque in bloques:          # cada bloque: (canales, muestras)
            limpio = filtro.procesar(bloque)

    Imagen: tfm_comun/imagenes/filtrado_filtro_continuo.png (de golpe o a trozos, el resultado es idéntico)
    """

    def __init__(self, fs):
        """Prepara el filtro para una señal a fs Hz. El estado se crea con el primer bloque que llegue."""
        self.sos = disenar_filtro(fs)
        self.estado = None

    def procesar(self, bloque):
        """Filtra el siguiente trozo de señal y guarda el estado para el trozo que venga después.

        Parámetros:
            bloque: matriz (canales x muestras) con la señal nueva, a la fs del constructor.

        Devuelve:
            Matriz (canales x muestras) filtrada. Encadenando todos los bloques se obtiene exactamente
            lo mismo que con filtrar() sobre la señal completa.
        """
        bloque = np.asarray(bloque, dtype=np.float64)
        if self.estado is None:
            self.estado = _estado_inicial(self.sos, bloque[:, 0])
        filtrado, self.estado = sosfilt(self.sos, bloque, axis=-1, zi=self.estado)
        return filtrado
