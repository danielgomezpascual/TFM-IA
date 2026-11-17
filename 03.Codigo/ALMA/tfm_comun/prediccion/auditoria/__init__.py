"""Auditoría de atajos: pistas que no son el EEG preictal pero que podrían delatar la clase de una ventana.
Lo usa fase_02/05_auditoria_atajos (y servirá en la Fase 3 para el experimento de control).

Qué es un atajo (shortcut learning, Geirhos et al. 2020)
=========================================================
Una red aprende lo más fácil que separe las clases, sea o no lo que buscamos. Si las ventanas
⏳ preictales salen casi siempre de un tipo de equipo, del principio de la grabación o con dos canales
a 0, la red puede acertar mirando ESO y no los cambios del EEG antes de una crisis. En eval saldría una
buena nota que no se sostiene con otros datos (como los modelos de radiografía que reconocían el
hospital por una marca en la placa, Zech et al. 2018).

Cómo se mide
============
Cada pista se convierte en un número por ventana (una "variable de atajo") y se mira cuánto separa
las clases con la AUROC: 0,5 = no separa nada; 1 (o 0) = separa perfectamente. Como una variable
puede separar "al revés" (p. ej. las ⏳ preictales salen ANTES en el fichero, AUROC < 0,5), se usa la
FUERZA = max(AUROC, 1 - AUROC), que va de 0,5 (nada) a 1 (todo), y aparte la dirección.

Como una pista puede separar con "forma" (p. ej. el primer minuto del fichero es casi solo 💤 y los minutos 2-5
casi solo ⏳), la fuerza se calcula sobre la variable CODIFICADA POR FRANJAS (codificar_por_franjas): cada franja
se sustituye por su proporción de ⏳ en train. Así se captan también relaciones que no son "más / menos".

Todo trabaja con tablas (sin leer la señal): es rápido y no depende del modelo que se entrene después.

Imagen: fase_02/img/que_es_un_atajo.png

Cada pieza en su fichero (un solo motivo para cambiar cada una):
    variables.py   -> qué pistas se vigilan (VARIABLES_ATAJO) y cómo se calculan para cada ventana.
    codificacion.py -> cómo se convierte una pista en un número que ordena las ventanas (por tasa y por franjas).
    fuerza.py      -> cuánto separa una pista las clases (fuerza, dentro del paciente y semáforo).
    sesiones.py    -> a qué sesión pertenece cada ventana y si esa sesión tiene crisis (control de tipo de fichero).
    emparejado.py  -> el conjunto de control con las 💤 en los mismos minutos del fichero que las ⏳.
"""



