"""
Formato de números legibles.

Motivo para cambiar: que cambie cómo se escriben los números
(separador de miles, decimales...).
"""


def numero_legible(valor):
    """Número en formato español: 1000 -> '1.000', 2.5 -> '2,5', 0.5 -> '0,5'.
    Los enteros llevan punto de miles; el resto, una cifra decimal (dos cifras significativas si es < 1)."""
    if valor >= 1 and float(valor).is_integer():
        return f'{valor:,.0f}'.replace(',', '.')
    if valor >= 1:
        return f'{valor:,.1f}'.replace(',', ' ').replace('.', ',').replace(' ', '.')
    return f'{valor:.2g}'.replace('.', ',')                 # 0,5 / 0,25 / 0,77
