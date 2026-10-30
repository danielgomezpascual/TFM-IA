"""
Formato de números y duraciones legibles.

Motivo para cambiar: que cambie cómo se escriben los números
(separador de miles, decimales...).
"""


def numero_legible(valor):
    """Escribe un número en formato español: punto para los miles y coma para los decimales.

    Reglas:
        - Enteros >= 1: con punto de miles y sin decimales.  1000 -> '1.000' · 7665 -> '7.665'
        - No enteros >= 1: una cifra decimal.                 2.5 -> '2,5' · 13.24 -> '13,2'
        - Menores que 1: dos cifras significativas.           0.5 -> '0,5' · 0.2213 -> '0,22'
    Truco: Python escribe 1,000.5; se cambia ',' por un espacio, '.' por ',' y el espacio por '.'.
    """
    if valor >= 1 and float(valor).is_integer():
        return f'{valor:,.0f}'.replace(',', '.')
    if valor >= 1:
        return f'{valor:,.1f}'.replace(',', ' ').replace('.', ',').replace(' ', '.')
    return f'{valor:.2g}'.replace('.', ',')                 # 0,5 / 0,25 / 0,77


def duracion_legible(segundos):
    """Escribe una duración en la unidad más cómoda: segundos si es menos de 1 minuto, si no, minutos.

    Ejemplos: 50 -> '50 s' · 240 -> '4 min' · 90 -> '1,5 min' · 300 -> '5 min'.
    Lo usa el esquema de periodos (esquemas.py) para escribir el preictal, el SPH y el margen postictal.
    """
    if abs(segundos) < 60:
        return f'{numero_legible(round(segundos))} s'
    return f'{numero_legible(round(segundos / 60, 1))} min'
