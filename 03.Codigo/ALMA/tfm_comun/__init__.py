"""
Utilidades comunes a todos los notebooks del TFM, ordenadas por funcionalidad.

Cada módulo tiene una sola responsabilidad (un solo motivo para cambiar):

entorno/      dónde están las cosas y con qué versiones
    rutas.py          -> dónde están el dataset y las salidas.
    versiones.py      -> versiones de Python y de las librerías de cada notebook.
    requisitos.py     -> qué librerías necesita cada notebook y si están instaladas (se comprueba al empezar).

dataset/      leer y entender TUSZ
    particiones.py    -> qué particiones tiene el dataset (train, dev, eval).
    edf.py            -> lectura de un EDF de TUSZ con MNE (entero o un trozo).
    anotaciones.py    -> lectura de las anotaciones .csv/.csv_bi y crisis de un fichero.
    canales.py        -> qué es cada canal del EDF (tipo, región, referencia).
    tipos_crisis.py   -> cada tipo de crisis: ficha clínica y grupo de inicio (ILAE).
    regiones_cerebro.py -> regiones de la corteza (atlas Desikan-Killiany): nombre en español y lóbulo.

senal/        transformar y guardar la señal (lo que se reutiliza en la demo)
    montaje.py        -> montaje bipolar TCP de 22 canales y canales de las figuras.
    remuestreo.py     -> remuestreo a la frecuencia común (256 Hz).
    filtrado.py       -> filtro paso banda 0,5-40 Hz y notch de 60 Hz (también a trozos, para la demo).
    preprocesado.py   -> orden de la cadena: montaje -> µV -> remuestreo -> filtrado.
    normalizacion.py  -> normalización por ventana y canal (z-score o robusta).
    calidad.py        -> indicadores de calidad de la señal (amplitud, canales planos).
    bandas.py         -> bandas de frecuencia del EEG (delta, theta, alfa, beta, gamma).
    senales.py        -> cómo se guarda y se carga la señal preprocesada (.npy en float16).
    fuentes.py        -> localización de fuentes (sLORETA sobre fsaverage): dónde se origina la crisis.

prediccion/   reglas para etiquetar y evaluar (aquí irán ventanas, modelos y métricas)
    umbrales.py       -> umbrales de predicción: lead-in, SPH y margen postictal de cada tipo.
    ventanas.py       -> ventanas de 4 s: cómo se cortan y a qué modelo pertenece cada una.
    balanceo.py       -> cuántas ventanas 💤 interictales se quedan de cada paciente (máximo y equilibrado).
    pliegues.py       -> reparto de pacientes en pliegues para la validación cruzada por paciente.
    atajos.py         -> pistas que no son EEG preictal (canales a 0, equipo, minuto del fichero...): medirlas y controlarlas.
    metricas.py       -> métricas de los modelos (AUROC con intervalo de confianza por paciente).

visual/       cómo se ven las cosas
    iconos.py         -> iconos de etiquetas, grupos, particiones, tipos de canal y regiones.
    colores.py        -> colores de las figuras (particiones, etiquetas, periodos, canales, regiones, bandas).
    numeros.py        -> cómo se escribe un número o una duración (formato español).
    ejes.py           -> cómo se muestran los ejes de las figuras.
    tablas.py         -> cómo se muestran las tablas (separadores entre particiones, formato).
    cabeza.py         -> dibujo de la cabeza con los electrodos 10-20 y el montaje.
    esquemas.py       -> esquema de los periodos alrededor de una crisis (preictal, SPH, postictal...).
    ficheros.py       -> cómo se enseña un fichero recién guardado (lo relee del disco y lo muestra como tabla).
    tipografia.py     -> letra de las figuras (Times).
    inicio_crisis.py  -> figura del inicio de la crisis estilo SEEG (trazas, cabeza y espectrograma).
    cerebro.py        -> cerebro 3D y vista desde arriba con el origen estimado de la crisis.

Se importa siempre desde el módulo concreto, para que se vea de dónde viene
cada cosa. Por ejemplo: from tfm_comun.visual.iconos import con_icono

imagenes/     figuras sintéticas que explican las funciones. Cada docstring que tiene una
              lleva una línea "Imagen: tfm_comun/imagenes/<nombre>.png". Índice con todas:
              tfm_comun/imagenes/README.md
"""
