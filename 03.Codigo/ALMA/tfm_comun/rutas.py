"""
Rutas del proyecto.

Motivo para cambiar: que cambie dónde están el dataset o las salidas.
Las rutas se calculan a partir de la ubicación de este fichero, así que
funcionan igual sea cual sea la carpeta desde la que se abre el notebook.

Estructura:
    D:\\Dani\\TFM IA\\                     <- RUTA_PROYECTO
        00.TUSZ\\v2.0.6\\edf\\              <- RUTA_EDF
        03.Codigo\\ALMA\\                   <- RUTA_ALMA
            tfm_comun\\rutas.py            (este fichero)
            fase_01\\                      <- RUTA_FASE_01: notebooks de la Fase 1
            outputs\\fase_01\\              <- RUTA_OUTPUTS_FASE_01: resultados de la Fase 1
                <notebook>\\figuras\\       <- carpeta_resultados('fase_01/<notebook>')
"""

from pathlib import Path

RUTA_ALMA = Path(__file__).resolve().parents[1]
RUTA_PROYECTO = RUTA_ALMA.parents[1]
RUTA_TUSZ = RUTA_PROYECTO / '00.TUSZ' / 'v2.0.6'
RUTA_EDF = RUTA_TUSZ / 'edf'
RUTA_OUTPUTS = RUTA_ALMA / 'outputs'

# Fase 1 (exploración de datos): notebooks y resultados
RUTA_FASE_01 = RUTA_ALMA / 'fase_01'
RUTA_OUTPUTS_FASE_01 = RUTA_OUTPUTS / 'fase_01'


def carpeta_resultados(nombre):
    """Crea (si no existe) outputs/<nombre>/figuras y devuelve (carpeta, carpeta_figuras)."""
    carpeta = RUTA_OUTPUTS / nombre
    figuras = carpeta / 'figuras'
    figuras.mkdir(parents=True, exist_ok=True)
    return carpeta, figuras


def buscar_edf(fichero):
    """Ruta del EDF a partir de su nombre ('aaaaacrb_s001_t001'), buscando en todas
    las particiones y sesiones: edf/<split>/<paciente>/<sesión>/<montaje>/<fichero>.edf"""
    paciente = fichero.split('_')[0]
    encontrados = list(RUTA_EDF.glob(f'*/{paciente}/*/*/{fichero}.edf'))
    assert len(encontrados) == 1, f'Se esperaba 1 EDF llamado {fichero}, hay {len(encontrados)}.'
    return encontrados[0]
