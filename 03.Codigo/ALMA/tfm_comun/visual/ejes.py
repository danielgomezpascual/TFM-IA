"""
Presentación de los ejes de las figuras.

Motivo para cambiar: que cambie cómo se muestran los ejes
(qué marcas se ponen, cuáles llevan texto...).
El formato de cada número no se decide aquí, sino en numeros.py.
"""

from matplotlib import ticker

from tfm_comun.visual.numeros import numero_legible


def eje_log_legible(ax, eje='y'):
    """Hace legible un eje con escala logarítmica: números normales en vez de potencias de 10.

    Qué hace:
        Pone marcas en 1, 2 y 5 de cada potencia de 10 (1, 2, 5, 10, 20, 50, 100, 200...) y las escribe
        en formato español con numero_legible (1.000, no 10³). Quita el texto de las marcas pequeñas.

    Parámetros:
        ax: eje de matplotlib que ya tiene escala log (ax.set_yscale('log')).
        eje: 'y' o 'x'.

    Imagen: tfm_comun/imagenes/ejes_eje_log_legible.png
    """
    eje_mpl = ax.yaxis if eje == 'y' else ax.xaxis
    eje_mpl.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    eje_mpl.set_major_formatter(ticker.FuncFormatter(lambda valor, _posicion: numero_legible(valor)))
    eje_mpl.set_minor_formatter(ticker.NullFormatter())
