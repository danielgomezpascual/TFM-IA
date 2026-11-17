"""Entrenar la red: el bucle de épocas con el que aprende una red de predicción (⏳ preictal frente a 💤 interictal).

Motivo para cambiar: que cambie cómo se entrena (optimizador, épocas, medidas por época, aumentos que se aplican).

Imágenes: fase_03/img/un_paso_de_entrenamiento.png (los 7 pasos de cada lote) y
          fase_03/img/parada_temprana.png (las curvas de aprendizaje y el momento de parar)

Cómo aprende, en sencillo
=========================
- Una ÉPOCA es pasar una vez por todas las filas de entrenamiento, en lotes de 128 (barajadas en cada época).
- Con cada lote, la red da sus logits, se mide el error con la PÉRDIDA (entropía cruzada binaria: castiga
  mucho estar muy seguro y equivocarse) y el optimizador AdamW mueve un poco todos los pesos en la dirección
  que reduce ese error. Lo "poco" lo marca la TASA DE APRENDIZAJE (lr).
- AdamW incluye el weight decay: empuja los pesos hacia 0 para que la red no dependa de unos pocos muy grandes
  (otra defensa contra el sobreajuste).

Cuándo se para: PARADA TEMPRANA (red/entreno/parada_temprana.py)
================================================================
Cada cierto tiempo se mide la AUC en 🟡 dev (por defecto, al final de cada época; con evaluaciones_por_epoca = 4,
cada cuarto de época). Se guarda la red de la MEJOR medida y, si en `paciencia` épocas no mejora, se para. Además,
si la AUC no mejora en 2 épocas, la tasa de aprendizaje se reduce a la mitad (pasos más finos cerca del mejor punto).

Por qué medir más de una vez por época (F03.05): en F03.04 la mejor red salía casi siempre en la época 1. Si el
mejor punto está a mitad de la primera época, midiendo solo al final se pierde.

Ajustes que añadió F03.05 (con su valor por defecto no cambian nada respecto a F03.04):
    evaluaciones_por_epoca (1) · apagar_canales (0) · tapar_frecuencias (0) · equilibrar_clases (False) ·
    criterio ('auc_global'): con qué nota se elige la mejor medida. 'auc_dentro_paciente' usa la AUC dentro del
    paciente (prediccion/evaluacion/auroc_paciente.py), que no premia separar pacientes entre sí; hace falta dar
    pacientes_valida. (criterio no está en ENTRENAMIENTO a propósito: así las configuraciones ya registradas en F03.05
    no cambian de huella.)

Ojo (para la memoria): elegir la época con 🟡 dev hace que la AUC de dev sea algo optimista. Es lo mismo que
pasó al elegir los hiperparámetros de XGBoost en F03.02. La nota limpia será la de 🟢 eval, al final de la fase.

Quién hace qué (cada pieza en su fichero)
=========================================
    el orden de las filas de cada época   -> red/lotes/orden_epoca.py
    los aumentos                          -> red/aumentos/
    puntuar la red y su pérdida media     -> red/entreno/puntuacion.py
    cuándo parar y qué pesos quedarse     -> red/entreno/parada_temprana.py
    guardar el punto de control           -> red/entreno/punto_control.py
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from tfm_comun.prediccion.evaluacion.auroc_paciente import auroc_dentro_paciente
from tfm_comun.red.aumentos.enmascarado import apagar_canales, tapar_frecuencias
from tfm_comun.red.entreno.parada_temprana import ParadaTemprana
from tfm_comun.red.entreno.punto_control import guardar_punto_control
from tfm_comun.red.entreno.puntuacion import perdida_media, puntuar_red
from tfm_comun.red.lotes.orden_epoca import orden_de_epoca


ENTRENAMIENTO = {'epocas_max': 40, 'paciencia': 6, 'tamano_lote': 128, 'lr': 1e-3, 'weight_decay': 1e-2, 'reflejo': 0.5,
                 'evaluaciones_por_epoca': 1, 'apagar_canales': 0.0, 'tapar_frecuencias': 0, 'equilibrar_clases': False}


def entrenar_red(modelo, datos, entrena, valida, k=None, semilla=42, pesos=None, grupos=None, ajustes=None, progreso=None,
                 pacientes_valida=None, lote_evaluacion=1024, punto_control=None):
    """Entrena la red con parada temprana sobre la AUC de validación.

    Parámetros:
        modelo: una red nueva (red/modelo.py: crear_red; cada tipo en red/redes/), ya en su dispositivo.
        datos: SecuenciasEspectro (red/lotes/secuencias_espectro.py).
        entrena, valida: posiciones (filas) de entrenamiento y de validación.
        k: ventanas de la secuencia que se usan (None = todas).
        semilla: fija el barajado, el reflejo y los aumentos (los pesos iniciales se fijan antes, al crear la red).
        pesos: un peso por fila de `entrena` para sortear las filas de cada época (None = todas una vez).
        grupos: un grupo por fila de `entrena`: en cada época entra una fila de cada grupo (red/lotes/orden_epoca.py:
            grupos_espaciados). None = todas.
        ajustes: diccionario que cambia algún valor de ENTRENAMIENTO (y, si se quiere, 'criterio', ver arriba).
        progreso: función tipo tqdm para la barra de épocas (opcional).
        pacientes_valida: el paciente de cada fila de `valida`. Si se da, el historial apunta también la AUC dentro
            del paciente; es obligatorio con criterio = 'auc_dentro_paciente'.
        lote_evaluacion: filas por lote al medir en validación. No cambia el resultado, solo la memoria de la GPU:
            la DCRNN de F03.08 necesita lotes más pequeños (con 1.024, cuBLAS fallaba en Windows).
        punto_control: ruta de un fichero .pt (opcional). Si se da, al final de cada época se guarda TODO lo necesario
            para seguir (pesos, optimizador, planificador, sorteos, historial, mejor época). Si el fichero ya existe,
            el entrenamiento SIGUE desde la época siguiente en vez de empezar de cero. Así, si se corta (un fallo de la
            GPU, un reinicio), solo se pierde la época en curso.

    Devuelve:
        (modelo, historial, mejor): el modelo con los pesos de la MEJOR medida; historial = tabla con una fila por
        medida (época, que puede ser fraccionaria: 1,25 = un cuarto de la segunda; pérdida de train desde la medida
        anterior, pérdida y AUC de validación, tasa de aprendizaje, segundos); mejor = {'epoca', 'auc_validacion' (la
        nota del criterio), 'criterio'}.

    Los pasos de cada lote (imagen: fase_03/img/un_paso_de_entrenamiento.png):
        1. lote -> 2. aumentos -> 3. la red da sus logits -> 4. pérdida -> 5. gradientes (backward) ->
        6. recorte de gradientes -> 7. el optimizador mueve los pesos.
    """
    ajustes = {**ENTRENAMIENTO, **(ajustes or {})}
    criterio = ajustes.get('criterio', 'auc_global')
    assert criterio in ('auc_global', 'auc_dentro_paciente'), f'Criterio desconocido: {criterio}'
    assert criterio == 'auc_global' or pacientes_valida is not None, 'Con criterio auc_dentro_paciente hay que dar pacientes_valida.'
    pacientes_valida = None if pacientes_valida is None else np.asarray(pacientes_valida)
    entrena, valida = np.asarray(entrena), np.asarray(valida)
    generador = torch.Generator().manual_seed(semilla)
    evaluaciones = max(1, int(ajustes['evaluaciones_por_epoca']))
    funcion_validacion = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=ajustes['lr'], weight_decay=ajustes['weight_decay'])
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='max', factor=0.5, patience=2 * evaluaciones)
    parada = ParadaTemprana(paciencia=ajustes['paciencia'] * evaluaciones)
    y_valida = datos.etiquetas[torch.as_tensor(valida).to(datos.dispositivo)].cpu().numpy()
    y_entrena = datos.etiquetas[torch.as_tensor(entrena).to(datos.dispositivo)].cpu().numpy()
    posicion_en_entrena = pd.Series(np.arange(len(entrena)), index=entrena)

    historial, epoca_inicial = [], 1
    if punto_control is not None and Path(punto_control).exists():     # seguir donde se dejó
        punto = torch.load(punto_control, map_location=datos.dispositivo, weights_only=False)
        modelo.load_state_dict(punto['modelo'])
        optimizador.load_state_dict(punto['optimizador'])
        planificador.load_state_dict(punto['planificador'])
        generador.set_state(punto['generador'])
        historial = punto['historial']
        parada.cargar(punto)
        epoca_inicial = punto['epoca'] + 1
    epocas = range(epoca_inicial, ajustes['epocas_max'] + 1) if not parada.parar else range(0)
    for epoca in (progreso(epocas) if progreso else epocas):
        orden = orden_de_epoca(entrena, generador, pesos, grupos)
        funcion = funcion_validacion
        if ajustes['equilibrar_clases']:                       # cada clase pesa lo mismo en la pérdida de esta época
            y_orden = y_entrena[posicion_en_entrena.loc[orden].to_numpy()]
            n_pos = max(float(y_orden.sum()), 1.0)
            funcion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_orden) - n_pos) / n_pos, device=datos.dispositivo))
        inicios = list(range(0, len(orden), ajustes['tamano_lote']))
        cortes = {int(round(len(inicios) * (j + 1) / evaluaciones)) for j in range(evaluaciones)}
        inicio, total, n = time.time(), 0.0, 0
        modelo.train()
        for numero, a in enumerate(inicios, start=1):
            # --- un paso de entrenamiento ---
            x, y = datos.lote(orden[a:a + ajustes['tamano_lote']], k=k, reflejo=ajustes['reflejo'], generador=generador)
            x = tapar_frecuencias(apagar_canales(x, ajustes['apagar_canales'], generador), ajustes['tapar_frecuencias'], generador)
            optimizador.zero_grad(set_to_none=True)
            perdida = funcion(modelo(x), y)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)   # evita pasos enormes si un lote es raro
            optimizador.step()
            total += perdida.item() * len(y)
            n += len(y)
            if numero not in cortes:
                continue
            # --- medida en validación ---
            p = puntuar_red(modelo, datos, valida, k=k, tamano_lote=lote_evaluacion)
            auc = float(roc_auc_score(y_valida, p))
            momento = epoca if evaluaciones == 1 else round(epoca - 1 + numero / len(inicios), 3)
            medida = {'época': momento, 'pérdida train': total / max(n, 1),
                      'pérdida validación': perdida_media(modelo, datos, valida, k, lote_evaluacion, funcion_validacion),
                      'AUC validación': auc, 'tasa de aprendizaje': optimizador.param_groups[0]['lr'], 'segundos': time.time() - inicio}
            if pacientes_valida is not None:
                medida['AUC dentro paciente'] = auroc_dentro_paciente(y_valida, p, pacientes_valida)
            historial.append(medida)
            nota = medida['AUC dentro paciente'] if criterio == 'auc_dentro_paciente' else auc
            planificador.step(nota)
            parada.actualizar(nota, modelo, momento)
            if parada.parar:
                break
            inicio, total, n = time.time(), 0.0, 0
            modelo.train()
        if punto_control is not None:
            guardar_punto_control(punto_control, {
                'modelo': modelo.state_dict(), 'optimizador': optimizador.state_dict(), 'planificador': planificador.state_dict(),
                'generador': generador.get_state(), 'historial': historial, 'epoca': epoca, **parada.estado()})
        if parada.parar:
            break
    modelo.load_state_dict(parada.mejor_estado)
    mejor = {'epoca': parada.mejor_momento, 'auc_validacion': parada.mejor_nota, 'criterio': criterio}
    return modelo, pd.DataFrame(historial).set_index('época'), mejor
