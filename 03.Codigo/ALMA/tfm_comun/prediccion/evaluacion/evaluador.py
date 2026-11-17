"""Evaluador: todas las notas de unas puntuaciones en las MISMAS ventanas, y la comparación emparejada de dos modelos.

Motivo para cambiar: que cambie qué notas se dan de cada modelo o cómo se comparan dos modelos.

Por qué una clase
=================
F03.05-F03.08 comparan muchos modelos (redes, XGBoost, variantes) siempre igual: con las mismas ventanas de 🟡 dev
(`mascara`), los mismos pacientes y los mismos sorteos del bootstrap. Antes cada notebook copiaba dos funciones
(medir y emparejada) que leían esas variables "de fuera". El Evaluador las recibe UNA vez al crearse y luego solo
hay que darle las puntuaciones de cada modelo: así no se puede comparar por error con otras ventanas.

Imagen: fase_03/img/comparacion_emparejada.png (por qué comparar dos modelos con los mismos sorteos de pacientes)
"""

import numpy as np

from tfm_comun.prediccion.evaluacion.auroc_global import auroc_con_intervalo, diferencia_auroc_con_intervalo
from tfm_comun.prediccion.evaluacion.auroc_paciente import (auroc_dentro_paciente_con_intervalo, auroc_por_paciente,
                                                            diferencia_auroc_dentro_paciente_con_intervalo)
from tfm_comun.prediccion.evaluacion.bootstrap import N_BOOTSTRAP
from tfm_comun.prediccion.evaluacion.sensibilidad import sensibilidad_con_especificidad


class Evaluador:
    """Mide y compara modelos con las mismas ventanas, los mismos pacientes y los mismos sorteos.

    Parámetros:
        y: etiqueta de cada fila (1 = ⏳ preictal, 0 = 💤 interictal).  pacientes: paciente de cada fila.
        mascara: qué filas se usan (p. ej. las de 🟡 dev que tienen puntuación de todos los modelos).
        especificidad: para la sensibilidad (0,9 = equivocarse como mucho en el 10 % de las 💤).
        n_bootstrap, semilla: los sorteos de pacientes de los intervalos (los mismos para todos los modelos).

    Ejemplo:
        evaluador = Evaluador(y, pacientes, comun, especificidad=0.9, n_bootstrap=500, semilla=42)
        notas_red = evaluador.notas(p_red)                       # diccionario con todas las notas
        diferencia = evaluador.diferencia(p_red, p_xgb)          # (A - B, inferior, superior), dentro del paciente
    """

    def __init__(self, y, pacientes, mascara, especificidad=0.9, n_bootstrap=N_BOOTSTRAP, semilla=42):
        self.y, self.pacientes = np.asarray(y), np.asarray(pacientes)
        self.mascara = np.asarray(mascara, dtype=bool)
        self.especificidad, self.n_bootstrap, self.semilla = especificidad, n_bootstrap, semilla

    def notas(self, p):
        """Las notas de unas puntuaciones (un número por fila; solo se usan las de la máscara).

        Devuelve un diccionario:
            'AUC dentro del paciente' y su intervalo ('_inferior_dentro', '_superior_dentro'): la que decide;
            'AUC-ROC' (global) y su intervalo ('_inferior', '_superior');
            'sensibilidad (esp. 90 %)' (con la especificidad dada) y 'AUC por paciente (mediana)'.
        Las claves que empiezan por '_' son para guardar o escribir el intervalo, no para enseñarlas en tablas.
        """
        m = self.mascara
        y, p, pacientes = self.y[m], np.asarray(p)[m], self.pacientes[m]
        auc, inferior, superior = auroc_con_intervalo(y, p, pacientes, n_bootstrap=self.n_bootstrap, semilla=self.semilla)
        dentro = auroc_dentro_paciente_con_intervalo(y, p, pacientes, n_bootstrap=self.n_bootstrap, semilla=self.semilla)
        pp = auroc_por_paciente(y, p, pacientes)
        return {'AUC dentro del paciente': dentro[0], '_inferior_dentro': dentro[1], '_superior_dentro': dentro[2],
                'AUC-ROC': auc, '_inferior': inferior, '_superior': superior,
                'sensibilidad (esp. 90 %)': sensibilidad_con_especificidad(y, p, self.especificidad)[0],
                'AUC por paciente (mediana)': float(pp.auroc.median()) if len(pp) else np.nan}

    def diferencia(self, a, b, dentro=True):
        """Diferencia emparejada A - B con su intervalo: (diferencia, inferior, superior).

        dentro=True (por defecto): con la AUC dentro del paciente. dentro=False: con la AUC global.
        Si el intervalo no incluye el 0, la diferencia es real con estos pacientes (lectura.py lo dice en palabras).
        """
        funcion = diferencia_auroc_dentro_paciente_con_intervalo if dentro else diferencia_auroc_con_intervalo
        m = self.mascara
        return funcion(self.y[m], np.asarray(a)[m], np.asarray(b)[m], self.pacientes[m],
                       n_bootstrap=self.n_bootstrap, semilla=self.semilla)
