"""
Rutas del proyecto.

Motivo para cambiar: que cambie dónde están el dataset o las salidas.
Las rutas se calculan a partir de la ubicación de este fichero, así que
funcionan igual sea cual sea la carpeta desde la que se abre el notebook.

Estructura:
    D:\\Dani\\TFM IA\\                     <- RUTA_PROYECTO
        00.TUSZ\\v2.0.6\\edf\\              <- RUTA_EDF
        03.Codigo\\ALMA\\                   <- RUTA_ALMA
            tfm_comun\\entorno\\rutas.py            (este fichero)
            fase_01\\                      <- RUTA_FASE_01: notebooks de la Fase 1
            fase_02\\                      <- RUTA_FASE_02: notebooks de la Fase 2
            outputs\\fase_01\\              <- RUTA_OUTPUTS_FASE_01: resultados de la Fase 1
                <notebook>\\figuras\\       <- carpeta_resultados('fase_01/<notebook>')
            outputs\\fase_02\\              <- RUTA_OUTPUTS_FASE_02: resultados de la Fase 2
                02_preprocesado\\senales\\  <- RUTA_SENALES: señal preprocesada (~36 GB, no subir a GitHub)
            outputs\\comun\\modelos_cabeza\\ <- RUTA_MODELOS_CABEZA: modelos directos guardados (senal/fuentes.py)
"""

from pathlib import Path

RUTA_ALMA = Path(__file__).resolve().parents[2]   # tfm_comun/entorno/rutas.py -> ALMA
RUTA_PROYECTO = RUTA_ALMA.parents[1]
RUTA_TUSZ = RUTA_PROYECTO / '00.TUSZ' / 'v2.0.6'
RUTA_EDF = RUTA_TUSZ / 'edf'
RUTA_OUTPUTS = RUTA_ALMA / 'outputs'

# Fase 1 (exploración de datos): notebooks y resultados
RUTA_FASE_01 = RUTA_ALMA / 'fase_01'
RUTA_OUTPUTS_FASE_01 = RUTA_OUTPUTS / 'fase_01'

# Fase 2 (preprocesado y ventanas): notebooks y resultados
RUTA_FASE_02 = RUTA_ALMA / 'fase_02'
RUTA_OUTPUTS_FASE_02 = RUTA_OUTPUTS / 'fase_02'

# Señal preprocesada (02_preprocesado): un .npy por tramo, en <split>/<clase>/
RUTA_SENALES = RUTA_OUTPUTS_FASE_02 / '02_preprocesado' / 'senales'

# Modelos directos de la localización de fuentes (uno por conjunto de electrodos, ~10 MB cada uno).
# Se calculan la primera vez y se reutilizan en todos los notebooks (tfm_comun/senal/fuentes.py)
RUTA_MODELOS_CABEZA = RUTA_OUTPUTS / 'comun' / 'modelos_cabeza'


def carpeta_resultados(nombre):
    """Prepara la carpeta de resultados de un notebook y su subcarpeta de figuras.

    Crea outputs/<nombre>/figuras si no existe (también las carpetas intermedias) y devuelve las dos rutas.
    Ejemplo: carpeta, figuras = carpeta_resultados('fase_01/03_visualizacion')
    """
    carpeta = RUTA_OUTPUTS / nombre
    figuras = carpeta / 'figuras'
    figuras.mkdir(parents=True, exist_ok=True)
    return carpeta, figuras


def buscar_edf(fichero):
    """Encuentra la ruta de UN EDF a partir de su nombre ('aaaaacrb_s001_t001').

    Qué hace:
        El nombre empieza por el paciente ('aaaaacrb'), así que solo busca dentro de su carpeta, en todas
        las particiones y sesiones: edf/<split>/<paciente>/<sesión>/<montaje>/<fichero>.edf.
        Si no hay exactamente uno, para con un error que dice cuántos ha encontrado.

    Para localizar miles de ficheros es mejor indice_edf(), que recorre el dataset una sola vez.
    """
    paciente = fichero.split('_')[0]
    encontrados = list(RUTA_EDF.glob(f'*/{paciente}/*/*/{fichero}.edf'))
    assert len(encontrados) == 1, f'Se esperaba 1 EDF llamado {fichero}, hay {len(encontrados)}.'
    return encontrados[0]


def indice_edf():
    """Diccionario {nombre del fichero: ruta del EDF} de TODO el dataset.

    Recorre las carpetas una sola vez (unos segundos) y después cada búsqueda es instantánea.
    Mucho más rápido que llamar miles de veces a buscar_edf(). Lo usa fase_02/02_preprocesado.
    Ejemplo: edf_de = indice_edf();  ruta = edf_de['aaaaacrb_s001_t001']
    """
    return {ruta.stem: ruta for ruta in RUTA_EDF.glob('*/*/*/*/*.edf')}
