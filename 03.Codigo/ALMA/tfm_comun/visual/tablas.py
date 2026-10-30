"""
Cómo se muestran las tablas en los notebooks.

Motivo para cambiar: que cambie el aspecto de las tablas (separadores, formato de los números, destacado).
Solo afecta a cómo se ven: los CSV se guardan siempre con la tabla original.

Imágenes: tfm_comun/imagenes/tablas_con_separadores.png y tfm_comun/imagenes/tablas_mostrar_destacado.png
"""

import re

from IPython.display import HTML, display
from pandas.io.formats.style import Styler

from tfm_comun.visual.colores import COLOR_DESTACADO, FONDO_DESTACADO

TEXTO_DESTACADO = '#1a1a1a'   # negro: se lee sobre el fondo dorado aunque el notebook use el tema oscuro

SEPARADOR = '2px solid #888'


def _cambios(etiquetas):
    """Posiciones donde cambia el primer nivel de las etiquetas (p. ej. de 'train' a 'dev').

    Ejemplo: [('train','crisis'), ('train','pacientes'), ('dev','crisis'), ('dev','pacientes')] -> [2]
    (la línea separadora se pone antes de la columna 2). Lo usa con_separadores().
    """
    claves = [e[0] if isinstance(e, tuple) else e for e in etiquetas]
    return [j for j in range(1, len(claves)) if claves[j] != claves[j - 1]]


def con_separadores(df, decimales=1):
    """Prepara una tabla para mostrarla en el notebook: separadores entre grupos y números en español.

    Qué hace:
        - Si las columnas tienen varios niveles (p. ej. partición > crisis/pacientes), pone una línea
          vertical gruesa donde cambia el primer nivel: train | dev | eval.
        - Si las filas tienen varios niveles, pone una línea horizontal donde cambia el primer nivel.
        - Escribe los números con punto de miles y coma decimal (1.811 · 2,5).

    Parámetros:
        df: la tabla (DataFrame). decimales: cifras decimales de los números no enteros.

    Devuelve:
        Un Styler de pandas (se muestra con display(); no sirve para guardar en CSV).

    Imagen: tfm_comun/imagenes/tablas_con_separadores.png
    """
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
    """Copia de la tabla con las columnas de varios niveles unidas en un solo nombre.

    ('train', 'crisis') -> 'train_crisis'. Un CSV no guarda bien columnas agrupadas, así que se aplanan
    antes de guardar. Si la tabla ya tiene un solo nivel, se devuelve tal cual.
    """
    if df.columns.nlevels == 1:
        return df
    return df.set_axis([union.join(str(n) for n in c) for c in df.columns], axis=1)


def _html_negro(tabla):
    """HTML de una tabla con el estilo escrito en cada celda (texto negro, fondo blanco, alineado a la izquierda).

    Va en cada celda, y no en un bloque <style>, porque algunos visores (PyCharm) ignoran los bloques de estilo
    y, con el tema oscuro, el texto saldría blanco sobre el fondo dorado. Si recibe una tabla con estilo
    (Styler), usa solo sus datos. Las celdas vacías se escriben como '—'. Lo usa mostrar_destacado().
    """
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
    Todo va en negro sobre fondo claro, con el estilo en cada elemento, para que se lea también con el tema oscuro.

    Parámetros:
        titulo: texto del encabezado (sale con 🏆 delante).
        *contenidos: textos (admiten HTML, p. ej. <b>) y tablas, en orden.

    Ejemplo:
        mostrar_destacado('tramos.csv: el dataset', '<b>7.665 filas</b>', tabla_columnas, tabla_ejemplo)

    Imagen: tfm_comun/imagenes/tablas_mostrar_destacado.png
    """
    texto = f'color:{TEXTO_DESTACADO}; margin:6px 0'
    partes = [f'<p style="{texto}">{c}</p>' if isinstance(c, str) else _html_negro(c) for c in contenidos]
    display(HTML(f'<div style="background:{FONDO_DESTACADO}; color:{TEXTO_DESTACADO}; border:2px solid {COLOR_DESTACADO}; '
                 f'border-radius:8px; padding:10px 14px; margin:8px 0">'
                 f'<div style="color:{TEXTO_DESTACADO}; font-weight:700; font-size:1.1em; margin-bottom:6px">🏆 {titulo}</div>'
                 + ''.join(partes) + '</div>'))
