"""
Presentación de los ejes de las figuras.

Motivo para cambiar: que cambie cómo se muestran los ejes
(qué marcas se ponen, cuáles llevan texto...).
El formato de cada número no se decide aquí, sino en numeros.py.
"""

from matplotlib import ticker

from tfm_comun.numeros import numero_legible


def eje_log_legible(ax, eje='y'):
    """En un eje con escala logarítmica, muestra números normales
    (1, 2, 5, 10, 20, 50, 100, 1.000...) en vez de potencias de 10 (10², 10³)."""
    eje_mpl = ax.yaxis if eje == 'y' else ax.xaxis
    eje_mpl.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    eje_mpl.set_major_formatter(ticker.FuncFormatter(lambda valor, _posicion: numero_legible(valor)))
    eje_mpl.set_minor_formatter(ticker.NullFormatter())
