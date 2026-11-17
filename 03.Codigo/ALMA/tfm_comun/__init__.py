"""
Utilidades comunes a todos los notebooks del TFM, ordenadas por funcionalidad.

Cada módulo tiene una sola responsabilidad (un solo motivo para cambiar):

entorno/      dónde están las cosas, con qué versiones y en qué máquina
    rutas.py          -> dónde están el dataset y las salidas.
    configuracion.py  -> leer el configuracion.json de un notebook anterior y comprobar que es del mismo conjunto.
    nombres_fichero.py -> convertir un texto ('🟡 dev', 'pliegue 2') en un nombre de fichero seguro.
    versiones.py      -> versiones de Python y de las librerías de cada notebook.
    requisitos.py     -> qué librerías necesita cada notebook y la tabla de comprobación (se llama al empezar).
    instalacion.py    -> instalar con pip lo que falte (y PyTorch con GPU si se puede).
    gpu_nvidia.py     -> ¿hay tarjeta NVIDIA?, ¿qué CUDA admite su driver?, ¿la ve PyTorch?
    dispositivo.py    -> dónde se hacen las cuentas de la red (GPU NVIDIA, GPU de Apple o procesador).

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

prediccion/   reglas para etiquetar, ventanas, modelos clásicos y cómo se evalúa
    umbrales.py       -> umbrales de predicción: lead-in, SPH y margen postictal de cada tipo.
    ventanas.py       -> ventanas de 4 s: cómo se cortan y a qué modelo pertenece cada una.
    balanceo.py       -> cuántas ventanas 💤 interictales se quedan de cada paciente (máximo y equilibrado).
    pliegues.py       -> reparto de pacientes en pliegues para la validación cruzada por paciente.
    vistas.py         -> qué ventanas se usan para entrenar, validar, evaluar y en cada control (Fase 3).
    estudio.py        -> el conjunto de la Fase 3 (un solo sitio para cambiarlo).
    caracteristicas.py -> de la ventana a 110 números: potencia relativa por banda y canal (modelos clásicos).
    almacen_caracteristicas.py -> guardar y reutilizar lo calculado de cada ventana.
    validacion.py     -> rondas de entrenar y medir (🟡 dev, o pliegues por paciente), sin tocar 🟢 eval.
    linea_base.py     -> línea base: elegir, entrenar y puntuar los modelos clásicos (regresión logística y XGBoost).
    normalizacion_paciente.py -> cada paciente frente a sí mismo (mediana y rango de sus ventanas, sin etiquetas).
    control_sesion.py -> los controles de sesión de F03.03 §10 (💤 de la misma sesión, del mismo fichero).
    modelos/          -> cada modelo clásico en su fichero (regresion_logistica.py, xgboost_arboles.py, arboles_histograma.py).
    evaluacion/       -> cómo se mide un modelo (antes metricas.py):
        bootstrap.py, auroc_global.py, auroc_paciente.py, sensibilidad.py,
        evaluador.py (Evaluador: todas las notas y la comparación emparejada), lectura.py (la diferencia en palabras).
    auditoria/        -> las pistas tramposas de F02.05 (antes atajos.py):
        variables.py, codificacion.py, fuerza.py, sesiones.py, emparejado.py.
    contextos/        -> el contexto de una ventana (antes contexto.py):
        duracion.py, disponibilidad.py, calculo.py, indices.py, resumen.py.

red/          PyTorch: cómo le llegan los datos a la red, las redes y su entrenamiento
    paquetes.py       -> las ventanas de un conjunto y una partición, leídas una vez y guardadas juntas (.npy).
    reproducibilidad.py -> semillas de Python, numpy y PyTorch.
    preparacion.py    -> de los paquetes a las secuencias listas para entrenar (una sola función para F03.05-F03.08).
    grafo.py          -> el grafo de correlación de un clip (F03.08).
    modelo.py         -> catálogo de redes y crear_red (elige la red por su 'tipo').
    ventanas/         -> ventanas sueltas en bruto para PyTorch (F03.01): dataset_ventanas.py, preparar_lote.py, cargador.py.
    entradas/         -> qué números recibe la red: rango_frecuencias.py, espectro_relativo.py, log_fft_segundos.py, bandas_segundos.py.
    aumentos/         -> aumento de datos: reflejo_hemisferios.py, enmascarado.py.
    lotes/            -> secuencias_espectro.py (SecuenciasEspectro) y orden_epoca.py (qué filas entran en cada época).
    entreno/          -> punto_control.py, puntuacion.py, parada_temprana.py (ParadaTemprana), entrenar_red.py,
                         variantes.py (EntrenadorVariantes).
    experimentos/     -> huella.py, registro.py (Registro), registro_parciales.py (RegistroParciales).
    autosupervisado/  -> el preentrenamiento de F03.07: corpus_construccion.py, corpus.py (Corpus), estadisticos_paciente.py,
                         lotes_corpus.py (LotesCorpus), perdida_futuro.py, preentrenar.py, transferencia_pesos.py.
    redes/            -> cada red y cada pieza de red en su fichero:
        cnn_gru.py (RedCNNGRU) + codificador_ventana.py (CodificadorVentana)            F03.04
        gru_segundos.py (RedGRUSegundos)                                                F03.06
        preentreno.py (RedPreentreno)                                                   F03.07
        dcrnn.py (RedDCRNN) + celda_dcgru.py (CeldaDCGRU) + convolucion_difusion.py     F03.08

visual/       cómo se ven las cosas
    iconos.py         -> iconos de etiquetas, grupos, particiones, tipos de canal y regiones.
    colores.py        -> colores de las figuras (particiones, etiquetas, periodos, canales, regiones, bandas).
    numeros.py        -> cómo se escribe un número, una duración, una nota con su intervalo o una diferencia con signo.
    ejes.py           -> cómo se muestran los ejes de las figuras.
    tablas.py         -> cómo se muestran las tablas (separadores entre particiones, formato).
    cabeza.py         -> dibujo de la cabeza con los electrodos 10-20 y el montaje.
    esquemas.py       -> esquema de los periodos alrededor de una crisis (preictal, SPH, postictal...).
    ficheros.py       -> cómo se enseña un fichero recién guardado (lo relee del disco y lo muestra como tabla).
    tipografia.py     -> letra de las figuras (Times).
    inicio_crisis.py  -> figura del inicio de la crisis estilo SEEG (trazas, cabeza y espectrograma).
    cerebro.py        -> cerebro 3D y vista desde arriba con el origen estimado de la crisis.

Reglas de diseño (SOLID, en sencillo)
=====================================
    - Una responsabilidad por fichero: cada uno dice al principio su "Motivo para cambiar" (y solo tiene uno).
    - Una clase por fichero (con sus constantes). Las piezas de una red están en su propio fichero.
    - Las clases reciben lo que necesitan al crearse (inyección de dependencias): el Evaluador recibe las ventanas
      y los pacientes; el EntrenadorVariantes, los datos, el registro y las semillas. No leen variables "de fuera".
    - Las carpetas agrupan los ficheros que cambian juntos (evaluacion/, auditoria/, entreno/...).

Se importa siempre desde el módulo concreto, para que se vea de dónde viene
cada cosa. Por ejemplo: from tfm_comun.visual.iconos import con_icono

imagenes/     figuras sintéticas que explican las funciones. Cada docstring que tiene una
              lleva una línea "Imagen: tfm_comun/imagenes/<nombre>.png". Índice con todas:
              tfm_comun/imagenes/README.md
"""
