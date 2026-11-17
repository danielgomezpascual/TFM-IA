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


# ── Notas de los modelos (AUC, diferencias) en los textos de los notebooks ─────────────────────────────────────────
# Antes cada notebook de la Fase 3 llevaba su copia de estas tres funciones; ahora están aquí, una sola vez.

def decimal(valor, cifras=2):
    """Un número con coma decimal y un número fijo de cifras, para los textos: 0.734 -> '0,73' · (0.0312, 3) -> '0,031'."""
    return f'{valor:.{cifras}f}'.replace('.', ',')


def con_intervalo(valor, inferior, superior, cifras=2):
    """Una nota con su intervalo de confianza: (0.734, 0.68, 0.78) -> '0,73 [0,68 – 0,78]'."""
    return f'{decimal(valor, cifras)} [{decimal(inferior, cifras)} – {decimal(superior, cifras)}]'


def con_signo(valor, cifras=3):
    """Una diferencia con su signo delante, para ver de un vistazo si sube o baja: 0.0312 -> '+0,031' · -0.02 -> '-0,020'."""
    return ('+' if valor >= 0 else '') + decimal(valor, cifras)
