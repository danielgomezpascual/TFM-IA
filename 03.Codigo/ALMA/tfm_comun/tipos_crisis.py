"""
Información clínica de cada tipo de crisis (para los textos y las figuras de los notebooks) y su
grupo de inicio.

Motivo para cambiar: que se añada un tipo de crisis o cambie lo que se sabe de él.
Los números entre corchetes [n] son las referencias de la bibliografía de los notebooks.
Para añadir un tipo nuevo basta con añadir su entrada a TIPOS_CRISIS.
'rango_tipico_hz' es None cuando el tipo no tiene una frecuencia característica única.

'grupo': grupo de inicio según la ILAE (Liga Internacional contra la Epilepsia, 2017): 'focal' (fnsz,
cpsz, spsz), 'generalizada' (gnsz, tcsz, tnsz) o 'ausencia' (absz, que es generalizada pero va aparte
por sus parámetros y su señal). Lo usan los modelos de predicción por grupo de la Fase 3: el modelo
principal es el general, con todas las crisis; los grupos son experimentos para ver si separar mejora.

Los umbrales de predicción (lead-in, SPH, margen postictal) NO están aquí: están en umbrales.py.

No descritos: 🔳 mysz (mioclónica), porque no tiene ninguna crisis en train con lead-in suficiente
(solo 2, de un mismo paciente), y cnsz/atsz, que no aparecen en TUSZ v2.0.6.
"""

import unicodedata

# Grupos de inicio (ILAE) y grupo de los tipos sin ficha
GRUPOS = ['focal', 'generalizada', 'ausencia']
GRUPO_TIPOS_SIN_FICHA = {'mysz': 'generalizada'}   # la mioclónica suele ser generalizada; sin_tipo no tiene grupo

TIPOS_CRISIS = {
    'absz': {
        'nombre': 'de ausencia',
        'clinica': (
            'Desconexión breve (unos 5–20 s): el paciente se queda con la mirada fija, sin responder, '
            'y vuelve de golpe sin recordarlo. Es típica de la infancia.'),
        'firma_eeg': (
            '**Complejos punta-onda generalizados a unos 3 Hz**: una **punta** (descarga rápida de muchas '
            'neuronas a la vez) seguida de una **onda lenta** (inhibición), unas 3 veces por segundo y **en '
            'todos los canales a la vez**. Se produce porque el circuito entre la corteza y el tálamo entra en '
            'una oscilación sincronizada [12, 13]. La frecuencia suele empezar algo más rápida (3–4 Hz) y '
            'frenarse hacia el final (2–2,5 Hz) [10].'),
        'rango_tipico_hz': (2.5, 4),
        'que_buscar': [
            'Durante: ondas grandes, rítmicas y sincronizadas en todos los canales a la vez.',
            'Inicio y final bruscos: el paciente "desconecta" y vuelve de golpe.',
        ],
        'nota_espectro': (
            'Los "3 Hz" son un valor típico, no exacto: la punta-onda se va frenando a lo largo de la crisis, '
            'y Welch promedia la crisis entera, así que el pico puede caer algo por debajo de 3 Hz. '
            'Además, con trozos de 2 s el espectro solo tiene valores cada 0,5 Hz.'),
        'literatura': (
            'La ILAE define la ausencia típica por la descarga punta-onda generalizada de unos 3 Hz [9]. '
            'En niños con epilepsia de ausencia infantil la frecuencia media ronda los 3 Hz y baja a lo largo '
            'de la crisis [10]. El origen es una oscilación del circuito tálamo-cortical [12, 13]. Para un '
            'modelo de **detección**, una firma tan regular es fácil de aprender.'),
        'filtro_trazado_hz': (1, 20),   # la punta-onda y sus armónicos caben de sobra
        'grupo': 'ausencia',   # generalizada no motora, pero aparte: parámetros y señal muy distintos
    },
    'tnsz': {
        'nombre': 'tónica',
        'clinica': (
            'Contracción muscular **sostenida** (rigidez) del tronco y las extremidades, de unos segundos '
            'a cerca de un minuto. Es frecuente durante el sueño y típica del síndrome de Lennox-Gastaut. '
            'Puede provocar caídas bruscas.'),
        'firma_eeg': (
            'Lo típico es un **aplanamiento generalizado** de la señal (electrodecremento) y/o una **actividad '
            'rápida generalizada de baja amplitud (unos 10–25 Hz)** que va creciendo [9, 18]. En la práctica, '
            'la contracción muscular mete mucho **artefacto** (músculo y movimiento) que puede tapar el EEG, '
            'sobre todo al final de la crisis.'),
        'rango_tipico_hz': (10, 25),
        'que_buscar': [
            'Inicio: cambio respecto al fondo en todos los canales (aplanamiento o actividad rítmica).',
            'Durante: actividad rítmica que va creciendo, cada vez más contaminada por músculo y movimiento.',
            'Final y después: artefactos grandes (movimiento, electrodos que se sueltan).',
        ],
        'nota_espectro': (
            'El rango típico (actividad rápida de 10–25 Hz) sale de la literatura, pero en EEG de superficie '
            'esa actividad es de baja amplitud y suele quedar tapada por los artefactos de la propia contracción: '
            'el **músculo** mete potencia en las frecuencias altas (> 25–30 Hz) y el **movimiento** en las muy '
            'bajas. Por eso la frecuencia característica puede caer en uno de los dos extremos. Si sale fuera del '
            'rango típico, conviene mirar la señal antes de sacar conclusiones.'),
        'literatura': (
            'La ILAE define la crisis tónica por la rigidez sostenida [9], y en el EEG se describen el '
            'aplanamiento y la actividad rápida generalizada, sobre todo en el síndrome de Lennox-Gastaut [18]. '
            'Para un modelo, las crisis tónicas son más difíciles que las ausencias: hay pocas (15 en train, '
            'casi todas de un mismo paciente) y el artefacto muscular se parece a la propia crisis.'),
        'filtro_trazado_hz': (1, 30),   # deja ver la actividad rápida, a costa de algo más de músculo
        'grupo': 'generalizada',   # empieza a la vez en los dos hemisferios
    },
    'fnsz': {
        'nombre': 'focal no específica',
        'clinica': (
            'Crisis que **empieza en una zona** de un hemisferio. TUSZ usa esta etiqueta cuando no se puede '
            'precisar más (por ejemplo, si el paciente mantuvo o no la consciencia). Los síntomas dependen de '
            'la zona: sacudidas de un brazo, sensaciones, automatismos, desconexión...'),
        'firma_eeg': (
            'Un **patrón ictal focal**: actividad rítmica (a menudo theta o alfa, a veces delta) o actividad '
            'rápida de bajo voltaje que aparece **en un grupo de canales vecinos** y va **evolucionando** en '
            'frecuencia, amplitud y extensión. Puede propagarse al resto de la cabeza [9]. No hay una '
            'frecuencia típica única.'),
        'rango_tipico_hz': None,
        'que_buscar': [
            'Inicio: cambio en unos pocos canales vecinos, no en todos a la vez.',
            'Durante: actividad rítmica que evoluciona (cambia de frecuencia y amplitud) y puede extenderse.',
            'Comparar los canales de un lado con los del otro: la asimetría es la pista de una crisis focal.',
        ],
        'nota_espectro': (
            'Las crisis focales no tienen una frecuencia típica única: depende de la zona y del paciente. '
            'Además, el espectro es la media de todos los canales, así que una crisis que solo afecta a unos '
            'pocos se diluye. Por eso el zoom de la izquierda usa los canales donde está anotada la crisis.'),
        'literatura': (
            'La ILAE clasifica las crisis por su inicio: focal o generalizado [9]. Es el tipo más frecuente de '
            'TUSZ con mucha diferencia, así que dominará el entrenamiento. Para un modelo es un reto porque cada '
            'paciente tiene su foco: la firma cambia de canales y de forma entre pacientes.'),
        'filtro_trazado_hz': (1, 30),
        'grupo': 'focal',   # empieza en una zona de un hemisferio
    },
    'gnsz': {
        'nombre': 'generalizada no específica',
        'clinica': (
            'Crisis que afecta a **los dos hemisferios desde el inicio**, pero que no encaja en un tipo '
            'generalizado concreto (ausencia, tónica, tónico-clónica...). Es una etiqueta "cajón de sastre".'),
        'firma_eeg': (
            'Cambio **simultáneo en todos los canales** desde el inicio. El patrón es variable: punta-onda, '
            'polipunta-onda, actividad rítmica lenta o rápida, o aplanamiento [9]. No hay una frecuencia '
            'típica única.'),
        'rango_tipico_hz': None,
        'que_buscar': [
            'Inicio: cambio a la vez en los dos hemisferios (todos los canales).',
            'Durante: el patrón puede ser lento o rápido; lo común es que sea simétrico.',
        ],
        'nota_espectro': (
            'Al ser un grupo heterogéneo, no hay una frecuencia típica: cada crisis gnsz puede parecerse a un '
            'tipo generalizado distinto.'),
        'literatura': (
            'La ILAE reserva las categorías "no específicas" para las crisis que no se pueden clasificar mejor '
            'con la información disponible [9]. Para un modelo de clasificación, gnsz puede confundirse con los '
            'tipos generalizados concretos (absz, tnsz, tcsz).'),
        'filtro_trazado_hz': (1, 30),
        'grupo': 'generalizada',   # empieza a la vez en los dos hemisferios
    },
    'cpsz': {
        'nombre': 'parcial compleja',
        'clinica': (
            'Crisis focal **con alteración de la consciencia**: el paciente parece despierto pero no responde, '
            'a menudo con automatismos (masticar, manipular objetos). Es la terminología antigua; la ILAE la '
            'llama hoy "focal con alteración de la consciencia". Muchas se originan en el lóbulo temporal.'),
        'firma_eeg': (
            'Patrón ictal **focal**, con frecuencia en canales **temporales**: actividad rítmica (típicamente '
            'theta) que aparece en un lado y puede extenderse [9]. Como en toda crisis focal, la frecuencia '
            'depende del foco y del paciente.'),
        'rango_tipico_hz': None,
        'que_buscar': [
            'Inicio: actividad rítmica en un grupo de canales, a menudo temporales (T3, T4, F7, F8...).',
            'Durante: evolución en frecuencia y amplitud, y posible extensión al otro lado.',
        ],
        'nota_espectro': (
            'No hay una frecuencia típica única. El espectro medio de los 22 canales diluye la actividad '
            'focal; el zoom de la izquierda usa los canales donde está anotada la crisis.'),
        'literatura': (
            'La clasificación ILAE de 2017 sustituyó "parcial compleja" por "focal con alteración de la '
            'consciencia" [9]. TUSZ mantiene la etiqueta antigua. Para un modelo, la frontera entre cpsz y '
            'fnsz es clínica (la consciencia), no siempre visible en el EEG.'),
        'filtro_trazado_hz': (1, 30),
        'grupo': 'focal',   # empieza en una zona de un hemisferio
    },
    'tcsz': {
        'nombre': 'tónico-clónica',
        'clinica': (
            'La crisis "convulsiva" clásica: pérdida de consciencia, **fase tónica** (rigidez) seguida de una '
            '**fase clónica** (sacudidas rítmicas que se van espaciando) y recuperación lenta, con confusión.'),
        'firma_eeg': (
            'En la fase tónica, **actividad rápida generalizada** que crece en amplitud; en la clónica, '
            '**descargas de polipuntas** separadas por pausas que se van alargando; al acabar, **aplanamiento** '
            'postictal [9]. En la práctica, el EEG de superficie queda muy tapado por el **artefacto muscular** '
            'y de movimiento.'),
        'rango_tipico_hz': None,
        'que_buscar': [
            'Inicio: actividad rápida en todos los canales, enseguida tapada por el músculo.',
            'Durante: ritmo de sacudidas que se va espaciando hacia el final.',
            'Después: aplanamiento de la señal (supresión postictal) o artefactos de movimiento.',
        ],
        'nota_espectro': (
            'La actividad cerebral cambia de fase a fase (rápida en la tónica, rítmica más lenta en la clónica), '
            'así que no hay una frecuencia única. El músculo mete mucha potencia en las frecuencias altas.'),
        'literatura': (
            'La ILAE la define por la secuencia tónica y clónica [9]. Para un modelo es fácil de **detectar** '
            '(la señal cambia muchísimo), pero buena parte de lo que aprende puede ser artefacto muscular, no '
            'EEG. En TUSZ hay pocas en train (31).'),
        'filtro_trazado_hz': (1, 30),
        'grupo': 'generalizada',   # empieza a la vez en los dos hemisferios
    },
    'spsz': {
        'nombre': 'parcial simple',
        'clinica': (
            'Crisis focal **con la consciencia conservada**: el paciente nota síntomas (sensaciones, '
            'sacudidas de una parte del cuerpo, alteraciones visuales...) y lo recuerda. Terminología antigua; '
            'la ILAE la llama hoy "focal con consciencia preservada".'),
        'firma_eeg': (
            'Cuando se ve, es un patrón **focal** en pocos canales. Pero como la zona afectada suele ser '
            'pequeña, **en muchas crisis parciales simples el EEG de superficie no muestra cambios claros** '
            '[19].'),
        'rango_tipico_hz': None,
        'que_buscar': [
            'Cambios sutiles en pocos canales vecinos; puede que no se vea nada evidente.',
            'Comparar con la señal de antes: la diferencia puede estar más en el ritmo que en la amplitud.',
        ],
        'nota_espectro': (
            'Si la crisis apenas cambia el EEG de superficie, su espectro se parecerá mucho al interictal. '
            'Eso no es un fallo del análisis: es una limitación conocida del EEG de cuero cabelludo.'),
        'literatura': (
            'La ILAE la clasifica como focal con consciencia preservada [9]. Se ha descrito que muchas no '
            'producen cambios visibles en el EEG de superficie [19]. En TUSZ hay muy pocas (8 en train, de un '
            'solo paciente): para un modelo, casi imposible de aprender como clase aparte.'),
        'filtro_trazado_hz': (1, 30),
        'grupo': 'focal',   # empieza en una zona de un hemisferio
    },
}


def info_tipo(tipo):
    """Ficha de un tipo de crisis. Error claro si aún no está descrito."""
    assert tipo in TIPOS_CRISIS, (f'El tipo {tipo!r} no está descrito en tfm_comun/tipos_crisis.py. '
                                  f'Descritos: {list(TIPOS_CRISIS)}')
    return TIPOS_CRISIS[tipo]


def nombre_para_fichero(tipo):
    """Nombre del tipo apto para un nombre de fichero: 'de ausencia' -> 'ausencia',
    'tónico-clónica' -> 'tonico_clonica', 'focal no específica' -> 'focal_no_especifica'."""
    nombre = info_tipo(tipo)['nombre'].removeprefix('de ')
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode()
    return nombre.replace('-', '_').replace(' ', '_')


def grupo(tipo):
    """Grupo de inicio del tipo ('focal', 'generalizada' o 'ausencia'), o None si no se sabe (sin_tipo)."""
    return TIPOS_CRISIS.get(tipo, {}).get('grupo', GRUPO_TIPOS_SIN_FICHA.get(tipo))
