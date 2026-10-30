"""
Cómo se muestra un fichero recién guardado.

Motivo para cambiar: que cambie cómo se enseña en los notebooks lo que se acaba de escribir en disco.

Norma del TFM: cada vez que un notebook guarda un fichero, lo vuelve a leer DEL DISCO y enseña unas filas
como tabla, con una frase de qué guarda. Así se ve exactamente lo que se ha escrito (no lo que había en
memoria) y el notebook explica sus propias salidas.

Uso, justo después de guardar:
    tabla.to_csv(RUTA_RESULTADOS / 'tramos.csv', index=False)
    mostrar_fichero(RUTA_RESULTADOS / 'tramos.csv', 'una fila por tramo de señal: fichero, inicio y fin')
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from tfm_comun.visual.iconos import tabla_con_iconos
from tfm_comun.visual.numeros import numero_legible

FILAS = 5                                                   # filas de ejemplo que se enseñan
COLUMNAS_CON_ICONO = ['split', 'clase', 'tipo', 'label', 'grupo']   # columnas cuyos valores llevan icono
LARGO_MAXIMO = 90                                           # caracteres por valor en los JSON (el resto, "…")


def _tamano(ruta):
    """Tamaño del fichero en formato legible: '820 B', '12,5 KB', '1,3 MB'."""
    tamano = ruta.stat().st_size
    for unidad, factor in [('GB', 1e9), ('MB', 1e6), ('KB', 1e3)]:
        if tamano >= factor:
            return f'{numero_legible(round(tamano / factor, 1))} {unidad}'
    return f'{tamano} B'


def _tabla_csv(ruta, filas):
    tabla = pd.read_csv(ruta)
    forma = f'{numero_legible(len(tabla))} filas × {tabla.shape[1]} columnas'
    columnas = [c for c in COLUMNAS_CON_ICONO if c in tabla.columns]
    return forma, f'las {min(filas, len(tabla))} primeras filas', tabla_con_iconos(tabla.head(filas), columnas)


def _tabla_json(ruta, filas):
    contenido = json.loads(ruta.read_text(encoding='utf-8'))
    def corto(valor):
        texto = valor if isinstance(valor, str) else json.dumps(valor, ensure_ascii=False)
        return texto if len(texto) <= LARGO_MAXIMO else texto[:LARGO_MAXIMO] + '…'
    tabla = pd.DataFrame({'valor': {clave: corto(valor) for clave, valor in contenido.items()}})
    tabla.index.name = 'clave'
    return f'{len(tabla)} claves', 'todas las claves (son pocas)', tabla.style.set_properties(**{'text-align': 'left'})


def _tabla_txt(ruta, filas):
    lineas = ruta.read_text(encoding='utf-8').splitlines()
    tabla = pd.DataFrame({'texto': [l for l in lineas if l.strip()][:filas]})
    tabla.index = tabla.index + 1
    tabla.index.name = 'línea'
    return f'{numero_legible(len(lineas))} líneas', f'las {len(tabla)} primeras líneas con texto', \
        tabla.style.set_properties(**{'text-align': 'left'})


def _tabla_npy(ruta, filas):
    matriz = np.load(ruta, mmap_mode='r')
    forma = ' × '.join(numero_legible(n) for n in matriz.shape) + f' ({matriz.dtype})'
    vista = np.asarray(matriz[..., :filas], dtype=np.float32).reshape(-1, min(filas, matriz.shape[-1]))
    tabla = pd.DataFrame(vista, columns=[f'[{i}]' for i in range(vista.shape[1])])
    return forma, f'los {vista.shape[1]} primeros valores de cada fila', tabla


_LECTORES = {'.csv': _tabla_csv, '.json': _tabla_json, '.txt': _tabla_txt, '.npy': _tabla_npy}


def mostrar_fichero(ruta, que_es, filas=FILAS):
    """Vuelve a leer del disco un fichero recién guardado y lo enseña como tabla.

    Parámetros:
        ruta: el fichero (.csv, .json, .txt o .npy).
        que_es: una frase que explica qué guarda (p. ej. 'una fila por crisis: fichero, inicio, fin y tipo').
        filas: cuántas filas de ejemplo enseñar (5 por defecto).

    Muestra una cabecera (nombre, tamaño, filas × columnas y qué es) y debajo la tabla. Los valores de
    split, clase, tipo, label y grupo llevan su icono, solo para mostrar: el fichero no se toca.
    """
    ruta = Path(ruta)
    lector = _LECTORES.get(ruta.suffix.lower())
    assert lector, f'mostrar_fichero no sabe leer {ruta.suffix}: añade un lector en tfm_comun/visual/ficheros.py'
    forma, que_se_ve, tabla = lector(ruta, filas)
    display(Markdown(f'💾 **Guardado `{ruta.name}`** · {_tamano(ruta)} · {forma}  \n'
                     f'*Qué guarda:* {que_es}. *Abajo:* {que_se_ve}.'))
    display(tabla)
