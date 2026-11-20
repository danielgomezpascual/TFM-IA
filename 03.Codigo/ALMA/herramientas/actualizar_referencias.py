"""
Actualizar referencias del código del TFM después de mover carpetas o renombrar cosas.

Motivo para cambiar: que haya una reorganización nueva (se añade un PASO a la lista PASOS).

Qué hace
========
Recorre el código de ALMA (notebooks de fase_0X/ y herramientas/, y los .py de tfm_comun/) y cambia, según
las reglas del paso elegido:
    - nombres:        identificadores que se renombran (RUTA_SENALES -> RUTA_SENALES_TUSZ). En el código y en
                      el texto de los notebooks (así los recuadros "📦 Variables..." quedan bien).
    - modulos:        módulos que se mueven (tfm_comun.dataset.edf -> tfm_comun.dataset.tusz.edf). Imports.
    - codigo:         expresiones regulares que solo se aplican al código (rutas escritas entre comillas).
    - salidas_json:   expresiones regulares para los configuracion.json que ya hay en outputs/ (rutas guardadas),
                      para no tener que volver a ejecutar los notebooks.
No toca: las salidas de las celdas, los README, la carpeta .venv ni el propio rutas.py (ese se cambia a mano).

Uso (desde la carpeta ALMA, con el Python del proyecto)
=======================================================
    .venv\\Scripts\\python herramientas\\actualizar_referencias.py --paso C
        -> SIMULACIÓN: enseña qué cambiaría en cada fichero, sin escribir nada.
    .venv\\Scripts\\python herramientas\\actualizar_referencias.py --paso C --aplicar
        -> aplica los cambios.
Se puede ejecutar varias veces: las reglas no vuelven a cambiar lo que ya está cambiado.
Antes de aplicar, conviene tener el código guardado en Git (commit) para poder deshacer.
"""

import argparse
import json
import re
import sys
from pathlib import Path

RUTA_ALMA = Path(__file__).resolve().parents[1]
RUTA_OUTPUTS = RUTA_ALMA / 'outputs'

# Ficheros que nunca se tocan
EXCLUIDOS = {
    RUTA_ALMA / 'tfm_comun' / 'entorno' / 'rutas.py',
    Path(__file__).resolve(),
}
CARPETAS_IGNORADAS = {'.venv', 'outputs', '.ipynb_checkpoints', '__pycache__', '.idea', '.claude'}

SEP_JSON = r'(?:\\\\|/)'   # separador de carpetas tal como queda escrito dentro de un .json: \\ o /


# ======================================================================================
# PASOS de la reorganización. Para una reorganización nueva, añadir aquí su paso.
# ======================================================================================
PASOS = {
    # Paso C (2026-11-20): notebooks y salidas de TUSZ en fase_0X/tusz/ y outputs/fase_0X/tusz/
    'C': {
        'descripcion': 'Rutas por corpus: constantes de TUSZ con _TUSZ y rutas "fase_0X/..." -> "fase_0X/tusz/..."',
        'nombres': {
            'RUTA_FASE_01': 'RUTA_FASE_01_TUSZ',
            'RUTA_FASE_02': 'RUTA_FASE_02_TUSZ',
            'RUTA_FASE_03': 'RUTA_FASE_03_TUSZ',
            'RUTA_OUTPUTS_FASE_01': 'RUTA_OUTPUTS_FASE_01_TUSZ',
            'RUTA_OUTPUTS_FASE_02': 'RUTA_OUTPUTS_FASE_02_TUSZ',
            'RUTA_OUTPUTS_FASE_03': 'RUTA_OUTPUTS_FASE_03_TUSZ',
            'RUTA_SENALES': 'RUTA_SENALES_TUSZ',
            'RUTA_PAQUETES': 'RUTA_PAQUETES_TUSZ',
            'RUTA_CARACTERISTICAS': 'RUTA_CARACTERISTICAS_TUSZ',
        },
        'modulos': {},
        'codigo': [
            # 'fase_03/{CONJUNTO}/...'  ->  'fase_03/tusz/{CONJUNTO}/...'   (también con f'' y "")
            (r"""(['"])fase_0(\d)/(?!tusz/|chbmit/|comun/)""", r'\1fase_0\2/tusz/'),
        ],
        'salidas_json': [
            # rutas absolutas: ...\outputs\fase_02\02_preprocesado  ->  ...\outputs\fase_02\tusz\02_preprocesado
            (rf'(outputs{SEP_JSON})(fase_0\d)({SEP_JSON})(?!tusz|chbmit|comun)', r'\1\2\3tusz\3'),
            # rutas relativas a outputs/: "fase_03/general/..."  ->  "fase_03/tusz/general/..."
            (r'"(fase_0\d)/(?!tusz/|chbmit/|comun/)', r'"\1/tusz/'),
            # el corpus: ...\00.TUSZ\v2.0.6  ->  ...\00.Corpus\TUSZ\v2.0.6
            (rf'00\.TUSZ({SEP_JSON})', r'00.Corpus\1TUSZ\1'),
        ],
    },
    # Paso D (pendiente): módulos de TUSZ a tfm_comun/dataset/tusz/. Se rellena cuando se haga.
    # 'D': {'descripcion': ..., 'nombres': {}, 'modulos': {'tfm_comun.dataset.edf': 'tfm_comun.dataset.tusz.edf', ...},
    #       'codigo': [], 'salidas_json': []},
}


# ======================================================================================
# Reglas -> funciones que cambian un texto
# ======================================================================================
def reglas_texto(paso):
    """Reglas que se aplican al código y al texto de los notebooks (nombres)."""
    reglas = []
    for viejo, nuevo in paso['nombres'].items():
        reglas.append((re.compile(rf'(?<![\w.]){re.escape(viejo)}\b'), nuevo))
    return reglas


def reglas_codigo(paso):
    """Reglas que solo se aplican al código: módulos y rutas entre comillas."""
    reglas = []
    for viejo, nuevo in paso['modulos'].items():
        reglas.append((re.compile(rf'\b{re.escape(viejo)}\b'), nuevo))
    reglas += [(re.compile(patron), reemplazo) for patron, reemplazo in paso['codigo']]
    return reglas


def aplicar(texto, reglas):
    """Aplica las reglas en orden. Devuelve el texto nuevo y cuántos cambios ha hecho."""
    total = 0
    for patron, reemplazo in reglas:
        texto, n = patron.subn(reemplazo, texto)
        total += n
    return texto, total


# ======================================================================================
# Ficheros
# ======================================================================================
def ficheros_codigo():
    """Notebooks y .py de ALMA, sin .venv, outputs ni los excluidos."""
    for ruta in sorted(RUTA_ALMA.rglob('*')):
        if ruta.suffix not in ('.py', '.ipynb') or not ruta.is_file():
            continue
        if CARPETAS_IGNORADAS & set(ruta.relative_to(RUTA_ALMA).parts):
            continue
        if ruta.resolve() in EXCLUIDOS:
            continue
        yield ruta


def cambiar_py(ruta, r_texto, r_codigo):
    texto = ruta.read_text(encoding='utf-8')
    nuevo, n1 = aplicar(texto, r_texto)
    nuevo, n2 = aplicar(nuevo, r_codigo)
    return texto, nuevo, n1 + n2


def cambiar_notebook(ruta, r_texto, r_codigo):
    """Cambia el 'source' de cada celda: nombres en todas, reglas de código solo en las de código."""
    original = ruta.read_text(encoding='utf-8')
    nb = json.loads(original)
    total = 0
    for celda in nb['cells']:
        fuente = ''.join(celda.get('source', []))
        nueva, n = aplicar(fuente, r_texto)
        if celda['cell_type'] == 'code':
            nueva, m = aplicar(nueva, r_codigo)
            n += m
        if n:
            celda['source'] = nueva.splitlines(keepends=True)
            total += n
    if not total:
        return original, original, 0
    return original, json.dumps(nb, indent=1, ensure_ascii=False) + '\n', total


def lineas_cambiadas(antes, despues, maximo=4):
    """Unas pocas líneas 'antes -> después' para la simulación."""
    a, d = antes.splitlines(), despues.splitlines()
    ejemplos = [(x.strip(), y.strip()) for x, y in zip(a, d) if x != y]
    return ejemplos[:maximo], len(ejemplos)


# ======================================================================================
# Programa
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--paso', required=True, choices=sorted(PASOS), help='paso de la reorganización')
    parser.add_argument('--aplicar', action='store_true', help='escribir los cambios (sin esto, solo simula)')
    args = parser.parse_args()
    paso = PASOS[args.paso]
    modo = 'APLICANDO' if args.aplicar else 'SIMULACIÓN (no se escribe nada; añade --aplicar)'

    print(f'Paso {args.paso}: {paso["descripcion"]}')
    print(f'Modo: {modo}\nALMA: {RUTA_ALMA}\n')

    r_texto, r_codigo = reglas_texto(paso), reglas_codigo(paso)
    r_json = [(re.compile(p), r) for p, r in paso['salidas_json']]

    # 1) Código: notebooks y .py
    print('=== Código ===')
    total_ficheros = total_cambios = 0
    for ruta in ficheros_codigo():
        if ruta.suffix == '.py':
            antes, despues, n = cambiar_py(ruta, r_texto, r_codigo)
        else:
            antes, despues, n = cambiar_notebook(ruta, r_texto, r_codigo)
        if not n:
            continue
        total_ficheros += 1
        total_cambios += n
        ejemplos, n_lineas = lineas_cambiadas(antes, despues)
        print(f'\n{ruta.relative_to(RUTA_ALMA)}  ({n} cambios)')
        for x, y in ejemplos:
            print(f'   - {x[:110]}\n   + {y[:110]}')
        if n_lineas > len(ejemplos):
            print(f'   ... y {n_lineas - len(ejemplos)} líneas más')
        if args.aplicar:
            ruta.write_text(despues, encoding='utf-8', newline='\n')
    print(f'\nCódigo: {total_cambios} cambios en {total_ficheros} ficheros.\n')

    # 2) Salidas: configuracion.json y demás .json guardados en outputs/
    print('=== Salidas (.json en outputs/) ===')
    total_json = 0
    for ruta in sorted(RUTA_OUTPUTS.rglob('*.json')) if r_json else []:
        texto = ruta.read_text(encoding='utf-8')
        nuevo, n = aplicar(texto, r_json)
        if not n:
            continue
        total_json += 1
        print(f'{ruta.relative_to(RUTA_ALMA)}  ({n} rutas)')
        if args.aplicar:
            ruta.write_text(nuevo, encoding='utf-8', newline='\n')
    print(f'\nSalidas: {total_json} ficheros .json con rutas actualizadas.')

    # 3) Aviso: otras salidas que guardan rutas viejas (CSV pequeños). Solo se avisa, no se cambian.
    viejas = re.compile(r'00\.TUSZ|outputs[\\/]fase_0\d[\\/](?!tusz|chbmit|comun)')
    avisos = [r for r in RUTA_OUTPUTS.rglob('*.csv')
              if r.stat().st_size < 20e6 and viejas.search(r.read_text(encoding='utf-8', errors='ignore'))]
    if avisos:
        print('\n[AVISO] Estos .csv guardan rutas con la estructura vieja (no se cambian; revisar si se usan):')
        for r in avisos:
            print(f'   {r.relative_to(RUTA_ALMA)}')

    if not args.aplicar:
        print('\nNada escrito. Si todo cuadra, repite con --aplicar.')


if __name__ == '__main__':
    sys.exit(main())
