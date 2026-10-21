"""
Formato de números legibles.

Motivo para cambiar: que cambie cómo se escriben los números
(separador de miles, decimales...).
"""


def numero_legible(valor):
    """Número en formato español: 1000 -> '1.000', 0.5 -> '0,5'."""
    if valor >= 1:
        return f'{valor:,.0f}'.replace(',', '.')
    return f'{valor:g}'.replace('.', ',')
