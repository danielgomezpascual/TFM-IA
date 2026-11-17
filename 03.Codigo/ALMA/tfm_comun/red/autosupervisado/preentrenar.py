"""Preentrenar (F03.07): el bucle de épocas de la tarea sin etiquetas "ver 60 s y predecir los 12 s siguientes".

Motivo para cambiar: que cambie cómo se entrena la tarea de preentrenamiento (épocas, optimizador, paciencia).

Imagen: fase_03/img/preentreno_autosupervisado.png (los dos pasos: preentrenar y ajustar)

La idea (Tang et al., ICLR 2022, §3.3)
======================================
Tarea sin etiquetas: ver 60 s de EEG y PREDECIR los 12 s siguientes (su log-FFT, segundo a segundo). Para hacerlo
bien, la red tiene que aprender cómo evoluciona el EEG: ritmos, artefactos, cambios de estado. Eso se aprende con
TODA la señal de 🔵 train (también los ~400 pacientes sin crisis), no solo con las ventanas etiquetadas.
Después, la parte que lee la señal se copia a la red de predicción (red/autosupervisado/transferencia_pesos.py) y se
AJUSTA con las ⏳/💤 (ajuste fino). En Tang, preentrenar subió la AUC de detección entre 4 y 8 puntos.

Se puede parar y seguir: al final de cada época se guarda un punto de control (red/entreno/punto_control.py).
"""

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from tfm_comun.red.autosupervisado.perdida_futuro import mae_con_mascara, perdida_media_futuro
from tfm_comun.red.entreno.punto_control import guardar_punto_control


PREENTRENO = {'ventanas_contexto': 15, 'ventanas_futuro': 3, 'epocas_max': 30, 'muestras_por_epoca': 100_000,
              'tamano_lote': 256, 'lr': 1e-3, 'weight_decay': 1e-2, 'paciencia': 5, 'fraccion_validacion': 0.1,
              'ventanas_estadisticos': 400}


def preentrenar(modelo, lotes, inicios_entrena, inicios_valida, carpeta, ajustes=None, semilla=42, progreso=None):
    """Entrena la tarea de predecir el futuro, con puntos de control para poder parar y seguir.

    Parámetros:
        modelo: RedPreentreno (red/redes/preentreno.py) ya en su dispositivo.  lotes: LotesCorpus.
        inicios_entrena / inicios_valida: dónde empiezan las secuencias de entrenamiento y de validación (pacientes
            distintos: la validación de esta tarea sale de pacientes de 🔵 train que no se usan para preentrenar).
        carpeta: dónde se guardan los puntos de control (ultimo.pt, mejor.pt) y el historial.
        ajustes: cambia valores de PREENTRENO.

    Devuelve:
        (historial, mejor): historial = tabla por época (pérdida de entrenamiento y de validación, minutos);
        mejor = {'epoca', 'perdida_validacion'}. Los pesos de la mejor época quedan en carpeta/'mejor.pt'.
    """
    ajustes = {**PREENTRENO, **(ajustes or {})}
    carpeta.mkdir(parents=True, exist_ok=True)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=ajustes['lr'], weight_decay=ajustes['weight_decay'])
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=2)
    historial, epoca_inicial, mejor = [], 1, {'epoca': 0, 'perdida_validacion': np.inf}
    ultimo = carpeta / 'ultimo.pt'
    if ultimo.exists():                                    # seguir donde se dejó
        punto = torch.load(ultimo, map_location=lotes.dispositivo, weights_only=False)
        modelo.load_state_dict(punto['modelo'])
        optimizador.load_state_dict(punto['optimizador'])
        planificador.load_state_dict(punto['planificador'])
        historial, mejor, epoca_inicial = punto['historial'], punto['mejor'], punto['epoca'] + 1
    generador = np.random.default_rng(semilla + epoca_inicial)
    validacion = np.asarray(inicios_valida)
    if len(validacion) > 20_000:                           # la misma muestra de validación en todas las épocas
        validacion = np.random.default_rng(semilla).choice(validacion, 20_000, replace=False)
    sin_mejorar = (historial[-1]['época'] - mejor['epoca']) if historial and mejor['epoca'] else 0   # épocas sin mejorar al seguir
    epocas = range(epoca_inicial, ajustes['epocas_max'] + 1)
    for epoca in (progreso(epocas, initial=epoca_inicial - 1, total=ajustes['epocas_max']) if progreso else epocas):
        if sin_mejorar >= ajustes['paciencia']:
            break
        inicio = time.time()
        modelo.train()
        elegidos = generador.choice(inicios_entrena, min(ajustes['muestras_por_epoca'], len(inicios_entrena)), replace=False)
        total, n = 0.0, 0
        for a in range(0, len(elegidos), ajustes['tamano_lote']):
            x, y, m = lotes.lote(elegidos[a:a + ajustes['tamano_lote']])
            optimizador.zero_grad(set_to_none=True)
            perdida = mae_con_mascara(modelo(x), y, m)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            optimizador.step()
            total += perdida.item() * len(x)
            n += len(x)
        perdida_val = perdida_media_futuro(modelo, lotes, validacion, 1024)
        planificador.step(perdida_val)
        historial.append({'época': epoca, 'pérdida train': total / max(n, 1), 'pérdida validación': perdida_val,
                          'tasa de aprendizaje': optimizador.param_groups[0]['lr'], 'minutos': (time.time() - inicio) / 60})
        if perdida_val < mejor['perdida_validacion'] - 1e-4:
            mejor, sin_mejorar = {'epoca': epoca, 'perdida_validacion': perdida_val}, 0
            guardar_punto_control(carpeta / 'mejor.pt', {'modelo': modelo.state_dict(), 'epoca': epoca, 'perdida_validacion': perdida_val})
        else:
            sin_mejorar += 1
        guardar_punto_control(ultimo, {'modelo': modelo.state_dict(), 'optimizador': optimizador.state_dict(),
                                       'planificador': planificador.state_dict(), 'epoca': epoca, 'historial': historial, 'mejor': mejor})
    pd.DataFrame(historial).to_csv(carpeta / 'historial.csv', index=False)
    return pd.DataFrame(historial), mejor
