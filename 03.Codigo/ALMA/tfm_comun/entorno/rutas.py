"""
Rutas del proyecto.

Motivo para cambiar: que cambie dónde están los corpus, los notebooks o las salidas.
Las rutas se calculan a partir de la ubicación de este fichero, así que
funcionan igual sea cual sea la carpeta desde la que se abre el notebook.

Los dos corpus se tratan igual: cada fase tiene una carpeta por corpus para los notebooks
y la misma estructura para las salidas.
    🏥 TUSZ     -> 'tusz'
    🧒 CHB-MIT  -> 'chbmit'

Estructura:
    D:\\Dani\\TFM IA\\                         <- RUTA_PROYECTO
        00.Corpus\\                            <- RUTA_CORPUS: datos crudos, solo lectura (no subir a GitHub)
            TUSZ\\v2.0.6\\                      <- RUTA_TUSZ
                edf\\                          <- RUTA_EDF
            CHB-MIT\\1.0.0\\                    <- RUTA_CHBMIT: chb01\\ ... chb24\\
        03.Codigo\\ALMA\\                       <- RUTA_ALMA
            tfm_comun\\entorno\\rutas.py                (este fichero)
            fase_0X\\tusz\\                     <- carpeta_notebooks(X, 'tusz'): notebooks solo de TUSZ
            fase_0X\\chbmit\\                   <- carpeta_notebooks(X, 'chbmit'): notebooks solo de CHB-MIT
            fase_0X\\comun\\                    <- carpeta_notebooks(X, 'comun'): notebooks con parámetro CORPUS
            outputs\\fase_0X\\<corpus>\\         <- carpeta_corpus(X, corpus): resultados de la fase para ese corpus
                <notebook>\\figuras\\           <- carpeta_resultados('fase_0X/<corpus>/<notebook>')
            outputs\\fase_02\\<corpus>\\02_preprocesado\\senales\\        <- ruta_senales(corpus) (~36 GB en TUSZ, no subir a GitHub)
            outputs\\fase_03\\<corpus>\\01_datos_red\\paquetes\\          <- ruta_paquetes(corpus) (~GB, no subir a GitHub)
            outputs\\fase_03\\<corpus>\\02_linea_base\\caracteristicas\\  <- ruta_caracteristicas(corpus)
            outputs\\comun\\modelos_cabeza\\     <- RUTA_MODELOS_CABEZA: modelos directos guardados (senal/fuentes.py)

Cómo usarlo:
    - Notebook de un solo corpus: las constantes con el corpus en el nombre (RUTA_OUTPUTS_FASE_02_TUSZ,
      RUTA_SENALES_TUSZ, RUTA_OUTPUTS_FASE_01_CHBMIT...). Así se ve a simple vista con qué corpus trabaja.
    - Notebook común (parámetro CORPUS): las funciones (carpeta_corpus(2, CORPUS), ruta_senales(CORPUS)...).
"""

from pathlib import Path

RUTA_ALMA = Path(__file__).resolve().parents[2]   # tfm_comun/entorno/rutas.py -> ALMA
RUTA_PROYECTO = RUTA_ALMA.parents[1]
RUTA_OUTPUTS = RUTA_ALMA / 'outputs'

# Corpus (datos crudos): un dataset por carpeta, con su versión. Nunca se escribe dentro
RUTA_CORPUS = RUTA_PROYECTO / '00.Corpus'
RUTA_TUSZ = RUTA_CORPUS / 'TUSZ' / 'v2.0.6'
RUTA_EDF = RUTA_TUSZ / 'edf'
RUTA_CHBMIT = RUTA_CORPUS / 'CHB-MIT' / '1.0.0'

# Nombre de cada corpus en las carpetas de notebooks y de salidas
CORPUS_TUSZ = 'tusz'
CORPUS_CHBMIT = 'chbmit'
CORPUS_DISPONIBLES = [CORPUS_TUSZ, CORPUS_CHBMIT]
CARPETA_COMUN = 'comun'   # notebooks comunes (parámetro CORPUS); no es un corpus


def _comprobar_corpus(corpus, admite_comun=False):
    validos = CORPUS_DISPONIBLES + ([CARPETA_COMUN] if admite_comun else [])
    assert corpus in validos, f'Corpus {corpus!r} desconocido. Usa uno de {validos}.'


def carpeta_notebooks(fase, corpus):
    """Carpeta de los notebooks de una fase para un corpus ('tusz', 'chbmit') o para los comunes ('comun').

    Ejemplo: carpeta_notebooks(1, 'tusz') -> ALMA/fase_01/tusz
    """
    _comprobar_corpus(corpus, admite_comun=True)
    return RUTA_ALMA / f'fase_{fase:02d}' / corpus


def carpeta_corpus(fase, corpus):
    """Carpeta de resultados de una fase para un corpus.

    Ejemplo: carpeta_corpus(2, 'chbmit') -> ALMA/outputs/fase_02/chbmit
    """
    _comprobar_corpus(corpus)
    return RUTA_OUTPUTS / f'fase_{fase:02d}' / corpus


def ruta_senales(corpus):
    """Señal preprocesada (F02.02): un .npy por tramo, en <split>/<clase>/."""
    return carpeta_corpus(2, corpus) / '02_preprocesado' / 'senales'


def ruta_paquetes(corpus):
    """Paquetes (F03.01): las ventanas de cada conjunto y partición, juntas en un .npy listo para la red."""
    return carpeta_corpus(3, corpus) / '01_datos_red' / 'paquetes'


def ruta_caracteristicas(corpus):
    """Características (F03.02): potencia relativa por banda y canal de cada ventana, para los modelos clásicos."""
    return carpeta_corpus(3, corpus) / '02_linea_base' / 'caracteristicas'


# 🏥 TUSZ: notebooks y resultados de cada fase
RUTA_FASE_01_TUSZ = carpeta_notebooks(1, CORPUS_TUSZ)
RUTA_FASE_02_TUSZ = carpeta_notebooks(2, CORPUS_TUSZ)
RUTA_FASE_03_TUSZ = carpeta_notebooks(3, CORPUS_TUSZ)
RUTA_OUTPUTS_FASE_01_TUSZ = carpeta_corpus(1, CORPUS_TUSZ)
RUTA_OUTPUTS_FASE_02_TUSZ = carpeta_corpus(2, CORPUS_TUSZ)
RUTA_OUTPUTS_FASE_03_TUSZ = carpeta_corpus(3, CORPUS_TUSZ)
RUTA_SENALES_TUSZ = ruta_senales(CORPUS_TUSZ)
RUTA_PAQUETES_TUSZ = ruta_paquetes(CORPUS_TUSZ)
RUTA_CARACTERISTICAS_TUSZ = ruta_caracteristicas(CORPUS_TUSZ)

# 🧒 CHB-MIT: notebooks y resultados de cada fase
RUTA_FASE_01_CHBMIT = carpeta_notebooks(1, CORPUS_CHBMIT)
RUTA_FASE_02_CHBMIT = carpeta_notebooks(2, CORPUS_CHBMIT)
RUTA_FASE_03_CHBMIT = carpeta_notebooks(3, CORPUS_CHBMIT)
RUTA_OUTPUTS_FASE_01_CHBMIT = carpeta_corpus(1, CORPUS_CHBMIT)
RUTA_OUTPUTS_FASE_02_CHBMIT = carpeta_corpus(2, CORPUS_CHBMIT)
RUTA_OUTPUTS_FASE_03_CHBMIT = carpeta_corpus(3, CORPUS_CHBMIT)
RUTA_SENALES_CHBMIT = ruta_senales(CORPUS_CHBMIT)
RUTA_PAQUETES_CHBMIT = ruta_paquetes(CORPUS_CHBMIT)
RUTA_CARACTERISTICAS_CHBMIT = ruta_caracteristicas(CORPUS_CHBMIT)

# Modelos directos de la localización de fuentes (uno por conjunto de electrodos, ~10 MB cada uno).
# Se calculan la primera vez y se reutilizan en todos los notebooks (tfm_comun/senal/fuentes.py)
RUTA_MODELOS_CABEZA = RUTA_OUTPUTS / 'comun' / 'modelos_cabeza'


def carpeta_resultados(nombre):
    """Prepara la carpeta de resultados de un notebook y su subcarpeta de figuras.

    Crea outputs/<nombre>/figuras si no existe (también las carpetas intermedias) y devuelve las dos rutas.
    Ejemplo: carpeta, figuras = carpeta_resultados('fase_01/tusz/03_visualizacion')
    """
    carpeta = RUTA_OUTPUTS / nombre
    figuras = carpeta / 'figuras'
    figuras.mkdir(parents=True, exist_ok=True)
    return carpeta, figuras


def buscar_edf(fichero):
    """Encuentra la ruta de UN EDF de 🏥 TUSZ a partir de su nombre ('aaaaacrb_s001_t001').

    Qué hace:
        El nombre empieza por el paciente ('aaaaacrb'), así que solo busca dentro de su carpeta, en todas
        las particiones y sesiones: edf/<split>/<paciente>/<sesión>/<montaje>/<fichero>.edf.
        Si no hay exactamente uno, para con un error que dice cuántos ha encontrado.

    Para localizar miles de ficheros es mejor indice_edf(), que recorre el dataset una sola vez.
    (Pasará al lector de TUSZ, tfm_comun/dataset/tusz/, en el paso D de la reorganización.)
    """
    paciente = fichero.split('_')[0]
    encontrados = list(RUTA_EDF.glob(f'*/{paciente}/*/*/{fichero}.edf'))
    assert len(encontrados) == 1, f'Se esperaba 1 EDF llamado {fichero}, hay {len(encontrados)}.'
    return encontrados[0]


def indice_edf():
    """Diccionario {nombre del fichero: ruta del EDF} de TODO 🏥 TUSZ.

    Recorre las carpetas una sola vez (unos segundos) y después cada búsqueda es instantánea.
    Mucho más rápido que llamar miles de veces a buscar_edf(). Lo usa F02.02.TUSZ (fase_02/tusz/02_preprocesado).
    Ejemplo: edf_de = indice_edf();  ruta = edf_de['aaaaacrb_s001_t001']
    """
    return {ruta.stem: ruta for ruta in RUTA_EDF.glob('*/*/*/*/*.edf')}
