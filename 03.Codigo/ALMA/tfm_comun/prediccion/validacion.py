"""
Validación: con qué ventanas se entrena y con cuáles se mide en cada ronda, sin tocar nunca 🟢 eval.

Motivo para cambiar: que cambie cómo se reparten las ventanas entre entrenar y validar.
Solo trabaja con tablas (las vistas de F03.01, prediccion/vistas.py); no sabe nada de modelos.

Dos casos (lo decide F02.04 para cada conjunto)
===============================================
- Conjunto con validación normal (p. ej. 'general'): UNA ronda. Se entrena con la vista *entrenamiento*
  (🔵 train equilibrado) y se mide en *validacion* (🟡 dev entero).
- Conjunto con validación cruzada por paciente (p. ej. 'grupo_ausencia', con pocos pacientes): una ronda por
  pliegue. En la ronda k se entrena con las ventanas de entrenamiento de los pliegues distintos de k y se mide
  en las de *validacion_cruzada* del pliegue k. Juntando todas las rondas, cada paciente se mide una vez con un
  modelo que no lo ha visto ("predicción fuera de pliegue").

En los dos casos el resultado es una puntuación por ventana de validación, medida siempre con un modelo que
no ha visto a ese paciente. 🟢 eval no aparece aquí: se usa una sola vez, al final de la Fase 3.
"""

import numpy as np


def rondas_validacion(filas):
    """Las rondas de entrenar y medir, como lista de (nombre, máscara de entrenamiento, máscara de validación).

    Parámetros:
        filas: tabla con una fila por ventana (las de los paquetes de 🔵 train y 🟡 dev juntas) y las columnas
            en_entrenamiento y en_validacion, o en_validacion_cruzada + pliegue (las deja F03.01).

    Devuelve:
        [(nombre, entrenar, validar)], con entrenar y validar arrays de True/False del largo de filas.
        Ejemplo con validación normal: [('🟡 dev', [...], [...])].
    """
    entrena = filas.en_entrenamiento.to_numpy(dtype=bool)
    if 'en_validacion' in filas and filas.en_validacion.any():
        return [('🟡 dev', entrena, filas.en_validacion.to_numpy(dtype=bool))]
    assert 'en_validacion_cruzada' in filas and 'pliegue' in filas, \
        'Las vistas no tienen ni validacion ni validacion_cruzada: revisa F03.01.'
    valida = filas.en_validacion_cruzada.to_numpy(dtype=bool)
    pliegue = filas.pliegue.to_numpy()
    rondas = []
    for k in sorted(np.unique(pliegue[valida & ~np.isnan(pliegue.astype(float))])):
        rondas.append((f'pliegue {int(k)}', entrena & (pliegue != k), valida & (pliegue == k)))
    return rondas
