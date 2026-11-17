"""
Control de sesión: medir un modelo quitando la pista de la GRABACIÓN (¿predice la crisis o reconoce la sesión?).

Motivo para cambiar: que cambie cómo se eligen las 💤 del control de sesión.
La usan F03.03 (§10, donde nació) y F03.04. Solo trabaja con tablas: no sabe nada de modelos.

Imagen: fase_03/img/control_sesion_esquema.png (tres grabaciones de un paciente; solo una con crisis)

Por qué
=======
F02.05 encontró (H5) que las ⏳ salen de las grabaciones con crisis y muchas 💤 de OTRAS grabaciones del mismo
paciente (otro día, otros electrodos). Un modelo podría acertar porque "esta grabación es distinta", sin saber
nada de crisis. El control compara las ⏳ solo con 💤 de las MISMAS sesiones (o ficheros) de su paciente:

    misma sesión   -> solo 💤 de las sesiones (_s00X) en las que ese paciente tiene alguna ⏳
    mismo fichero  -> solo 💤 de los ficheros (_t00X) en los que ese paciente tiene alguna ⏳ (control muy duro:
                      muchas de esas 💤 son de después de la crisis)
"""

import numpy as np
import pandas as pd


def sesion_de_fichero(ficheros):
    """La sesión de cada fichero: 'aaaaacrb_s001_t002' -> 'aaaaacrb_s001' (array de str)."""
    ficheros = pd.Series(np.asarray(ficheros)).astype(str)
    return ficheros.str.extract(r'^(.*?_s\d+)')[0].fillna(ficheros).to_numpy()


def mascara_mismo_grupo(validar, etiqueta, pacientes, grupo):
    """Las ⏳ de validación + solo las 💤 de validación de los grupos (sesión o fichero) donde su paciente tiene ⏳.

    Parámetros:
        validar: True en las filas de validación.  etiqueta: 1 = ⏳, 0 = 💤.  pacientes: paciente de cada fila.
        grupo: la sesión (sesion_de_fichero) o el fichero de cada fila.

    Devuelve:
        Array True/False del largo de las filas.
    """
    validar, etiqueta = np.asarray(validar, dtype=bool), np.asarray(etiqueta)
    pacientes, grupo = np.asarray(pacientes), np.asarray(grupo)
    con_preictal = set(zip(pacientes[validar & (etiqueta == 1)], grupo[validar & (etiqueta == 1)]))
    mismo = np.fromiter(((p, g) in con_preictal for p, g in zip(pacientes, grupo)), dtype=bool, count=len(pacientes))
    return validar & ((etiqueta == 1) | ((etiqueta == 0) & mismo))


def controles_de_sesion(validar, etiqueta, pacientes, ficheros):
    """Los tres grupos de 💤 que se comparan: {'todas las 💤': ..., 'misma sesión': ..., 'mismo fichero': ...}."""
    return {'todas las 💤': np.asarray(validar, dtype=bool),
            'misma sesión': mascara_mismo_grupo(validar, etiqueta, pacientes, sesion_de_fichero(ficheros)),
            'mismo fichero': mascara_mismo_grupo(validar, etiqueta, pacientes, np.asarray(ficheros).astype(str))}
