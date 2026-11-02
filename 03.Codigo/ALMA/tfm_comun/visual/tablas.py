"""
Cómo se muestran las tablas en los notebooks.

Motivo para cambiar: que cambie el aspecto de las tablas (separadores, formato de los números, destacado).
Solo afecta a cómo se ven: los CSV se guardan siempre con la tabla original.

Imágenes: tfm_comun/imagenes/tablas_con_separadores.png y tfm_comun/imagenes/tablas_mostrar_destacado.png
"""

import re

import pandas as pd
from IPython.display import HTML, display
from pandas.io.formats.style import Styler

from tfm_comun.visual.colores import COLOR_DESTACADO, FONDO_DESTACADO

TEXTO_DESTACADO = '#1a1a1a'   # negro: se lee sobre el fondo dorado aunque el notebook use el tema oscuro

SEPARADOR = '2px solid #888'


PARTICION = re.compile(r'\b(train|dev|eval|todas|total)\b')   # columnas de partición, con o sin icono ('🔵 train'), y sus totales


def _cambios(etiquetas):
    """Posiciones donde cambia el primer nivel de las etiquetas (p. ej. de 'train' a 'dev').

    Ejemplo: [('train','crisis'), ('train','pacientes'), ('dev','crisis'), ('dev','pacientes')] -> [2]
    (la línea separadora se pone antes de la columna 2). Lo usa _separadores().
    """
    claves = [e[0] if isinstance(e, tuple) else e for e in etiquetas]
    return [j for j in range(1, len(claves)) if claves[j] != claves[j - 1]]


def _separadores(df):
    """Dónde van las líneas de una tabla: (columnas con línea a la izquierda, filas con línea encima).

    Reglas (las mismas para con_separadores() y para las tablas de mostrar_destacado()):
        - Columnas de varios niveles: donde cambia el primer nivel (train | dev | eval).
        - Columnas de un nivel: entre las columnas que son particiones o totales (🔵 train | 🟡 dev | 🟢 eval | ⚪ todas).
        - Filas de varios niveles: donde cambia el primer nivel.
        - Filas de un nivel con valores seguidos repetidos (p. ej. 3 filas de ejemplo de cada clase):
          donde cambia el valor (preictal | interictal | ictal).
    """
    if df.columns.nlevels > 1:
        columnas = _cambios(df.columns)
    else:
        columnas = [j for j, c in enumerate(df.columns) if j > 0 and PARTICION.search(str(c))]
    filas = []
    if df.index.nlevels > 1 or (len(df.index) > 1 and (df.index[1:] == df.index[:-1]).any()):
        filas = _cambios(df.index)
    return columnas, filas


def con_separadores(df, decimales=1):
    """Prepara una tabla para mostrarla en el notebook: separadores entre grupos y números en español.

    Qué hace:
        - Pone líneas verticales gruesas entre los grupos de columnas: train | dev | eval (con columnas de
          uno o de varios niveles).
        - Pone líneas horizontales entre los grupos de filas: cuando las filas tienen varios niveles, o
          cuando hay filas seguidas con el mismo valor (p. ej. ejemplos de cada clase).
        - Escribe los números con punto de miles y coma decimal (1.811 · 2,5).
        Las reglas exactas están en _separadores().

    Parámetros:
        df: la tabla (DataFrame). decimales: cifras decimales de los números no enteros.

    Devuelve:
        Un Styler de pandas (se muestra con display(); no sirve para guardar en CSV).

    Imagen: tfm_comun/imagenes/tablas_con_separadores.png
    """
    columnas, filas = _separadores(df)
    estilos = ([{'selector': f'td.col{j}, th.col_heading.col{j}', 'props': f'border-left: {SEPARADOR}'} for j in columnas]
               + [{'selector': f'td.row{i}, th.row_heading.row{i}', 'props': f'border-top: {SEPARADOR}'} for i in filas])
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


def _spans(etiqueta):
    """Los atributos colspan y rowspan de una etiqueta HTML (los únicos que se conservan en _html_negro)."""
    return ''.join(' ' + a for a in re.findall(r'(?:colspan|rowspan)="\d+"', etiqueta.split('>')[0]))


def _numero_espanol(valor, decimales=2):
    """Un número con punto de miles y coma decimal: 2274 -> '2.274', 13.2 -> '13,2'. Lo vacío se deja vacío.
    Es el mismo formato que con_separadores(), para las tablas de mostrar_destacado()."""
    if pd.isna(valor):
        return valor
    texto = f'{valor:,.0f}' if float(valor).is_integer() else f'{valor:,.{decimales}f}'.rstrip('0')
    return texto.replace(',', '_').replace('.', ',').replace('_', '.')


def _html_negro(tabla):
    """HTML de una tabla con el estilo escrito en cada celda (texto negro, fondo blanco, alineado a la izquierda).

    Va en cada celda, y no en un bloque <style>, porque algunos visores (PyCharm) ignoran los bloques de estilo
    y, con el tema oscuro, el texto saldría blanco sobre el fondo dorado. Si recibe una tabla con estilo
    (Styler), usa solo sus datos. Las celdas vacías se escriben como '—'. Lleva las mismas líneas separadoras
    que con_separadores() (train | dev | eval, grupos de filas), también escritas en cada celda, y los números
    en formato español (2.274 · 13,2). Lo usa mostrar_destacado().
    """
    df = tabla.data if isinstance(tabla, Styler) else tabla   # de una tabla con estilo se usan solo los datos
    columnas, filas = _separadores(df)                         # las mismas líneas que con_separadores()
    df = df.astype(object)                                     # copia en la que caben los números ya escritos
    for j, tipo in enumerate(tabla.data.dtypes if isinstance(tabla, Styler) else tabla.dtypes):
        if pd.api.types.is_numeric_dtype(tipo) and not pd.api.types.is_bool_dtype(tipo):
            df.iloc[:, j] = df.iloc[:, j].map(_numero_espanol)
    celda = f'color:{TEXTO_DESTACADO}; background:#ffffff; padding:3px 8px; border-bottom:1px solid #eadfb8; text-align:left'
    vertical, horizontal = f'; border-left:{SEPARADOR}', f'; border-top:{SEPARADOR}'

    def cabecera(fila):
        """Fila de la cabecera: línea a la izquierda de las celdas que empiezan en una columna separada.
        Se recorre desde la derecha, sumando los colspan, porque a la izquierda están las celdas del índice."""
        celdas = re.findall(r'<th[^>]*>.*?</th>', fila, re.S)
        posicion, nuevas = len(df.columns), []
        for c in reversed(celdas):
            ancho = int(re.search(r'colspan="(\d+)"', c).group(1)) if 'colspan' in c else 1
            posicion -= ancho
            nuevas.append(f'<th{_spans(c)} style="{celda}; font-weight:700{vertical if posicion in columnas else ""}">'
                          + re.sub(r'^<th[^>]*>', '', c))
        return '<tr>' + ''.join(reversed(nuevas)) + '</tr>'

    def cuerpo(fila, i):
        """Fila de datos: línea encima si empieza un grupo de filas y a la izquierda de las columnas separadas."""
        arriba = horizontal if i in filas else ''
        fila = re.sub(r'<th[^>]*>', lambda m: f'<th{_spans(m.group(0))} style="{celda}; font-weight:700{arriba}">', fila)
        contador = iter(range(len(df.columns)))
        return re.sub(r'<td[^>]*>', lambda m: f'<td style="{celda}{arriba}{vertical if next(contador) in columnas else ""}">', fila)

    html = df.to_html(border=0, na_rep='—')
    html = re.sub(r'<table[^>]*>', f'<table style="border-collapse:collapse; margin:4px 0 10px; color:{TEXTO_DESTACADO}">', html)
    html = re.sub(r'<thead[^>]*>(.*?)</thead>',
                  lambda m: '<thead>' + ''.join(cabecera(f) for f in re.findall(r'<tr[^>]*>.*?</tr>', m.group(1), re.S)) + '</thead>',
                  html, flags=re.S)
    html = re.sub(r'<tbody[^>]*>(.*?)</tbody>',
                  lambda m: '<tbody>' + ''.join(cuerpo(f, i) for i, f in enumerate(re.findall(r'<tr[^>]*>.*?</tr>', m.group(1), re.S)))
                  + '</tbody>', html, flags=re.S)
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
