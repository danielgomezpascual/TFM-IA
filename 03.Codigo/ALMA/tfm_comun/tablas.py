"""
Cómo se muestran las tablas en los notebooks.

Motivo para cambiar: que cambie el aspecto de las tablas (separadores, formato de los números, destacado).
Solo afecta a cómo se ven: los CSV se guardan siempre con la tabla original.
"""

import re

from IPython.display import HTML, display
from pandas.io.formats.style import Styler

from tfm_comun.colores import COLOR_DESTACADO, FONDO_DESTACADO

TEXTO_DESTACADO = '#1a1a1a'   # negro: se lee sobre el fondo dorado aunque el notebook use el tema oscuro

SEPARADOR = '2px solid #888'


def _cambios(etiquetas):
    """Posiciones donde cambia el primer nivel de las etiquetas (p. ej. de 'train' a 'dev')."""
    claves = [e[0] if isinstance(e, tuple) else e for e in etiquetas]
    return [j for j in range(1, len(claves)) if claves[j] != claves[j - 1]]


def con_separadores(df, decimales=1):
    """Tabla con una línea vertical entre los grupos de columnas (p. ej. train | dev | eval) y una
    horizontal entre los grupos de filas, cuando las columnas o las filas tienen varios niveles.
    Los números van en formato español (1.811 · 2,5)."""
    estilos = []
    if df.columns.nlevels > 1:
        estilos += [{'selector': f'td.col{j}, th.col_heading.col{j}', 'props': f'border-left: {SEPARADOR}'}
                    for j in _cambios(df.columns)]
    if df.index.nlevels > 1:
        estilos += [{'selector': f'td.row{i}, th.row_heading.row{i}', 'props': f'border-top: {SEPARADOR}'}
                    for i in _cambios(df.index)]
    return (df.style.set_table_styles(estilos, overwrite=False)
            .format(precision=decimales, thousands='.', decimal=','))


def columnas_planas(df, union='_'):
    """Copia con las columnas de varios niveles unidas en un solo nombre ('train', 'crisis') -> 'train_crisis'.
    Para guardar en CSV tablas que se muestran con columnas agrupadas."""
    if df.columns.nlevels == 1:
        return df
    return df.set_axis([union.join(str(n) for n in c) for c in df.columns], axis=1)


def _html_negro(tabla):
    """HTML de una tabla con el estilo puesto en cada celda (texto negro, fondo claro, alineado a la izquierda).
    Va en cada celda, y no en un bloque <style>, porque algunos visores (PyCharm) ignoran los bloques de estilo."""
    df = tabla.data if isinstance(tabla, Styler) else tabla   # de una tabla con estilo se usan solo los datos
    html = df.to_html(border=0, na_rep='—')
    celda = f'color:{TEXTO_DESTACADO}; background:#ffffff; padding:3px 8px; border-bottom:1px solid #eadfb8; text-align:left'
    html = re.sub(r'<table[^>]*>', f'<table style="border-collapse:collapse; margin:4px 0 10px; color:{TEXTO_DESTACADO}">', html)
    html = re.sub(r'<th(\s[^>]*)?>', lambda m: f'<th style="{celda}; font-weight:700">', html)
    html = re.sub(r'<td(\s[^>]*)?>', lambda m: f'<td style="{celda}">', html)
    html = re.sub(r'<tr[^>]*>', '<tr>', html)
    html = re.sub(r'<thead[^>]*>', '<thead>', html)
    return html


def mostrar_destacado(titulo, *contenidos):
    """Muestra el resultado más importante de un notebook dentro de un recuadro dorado.
    contenidos: tablas (DataFrame o tabla con estilo) o textos, en el orden en que se muestran.
    Todo va en negro sobre fondo claro, con el estilo en cada elemento, para que se lea también con el tema oscuro."""
    texto = f'color:{TEXTO_DESTACADO}; margin:6px 0'
    partes = [f'<p style="{texto}">{c}</p>' if isinstance(c, str) else _html_negro(c) for c in contenidos]
    display(HTML(f'<div style="background:{FONDO_DESTACADO}; color:{TEXTO_DESTACADO}; border:2px solid {COLOR_DESTACADO}; '
                 f'border-radius:8px; padding:10px 14px; margin:8px 0">'
                 f'<div style="color:{TEXTO_DESTACADO}; font-weight:700; font-size:1.1em; margin-bottom:6px">🏆 {titulo}</div>'
                 + ''.join(partes) + '</div>'))
