"""
Predicción: reglas para etiquetar y evaluar (lead-in, SPH, margen postictal...), ventanas, modelos clásicos y métricas.

Ficheros sueltos: umbrales.py, ventanas.py, balanceo.py, pliegues.py, vistas.py, estudio.py, caracteristicas.py,
                  almacen_caracteristicas.py, validacion.py, linea_base.py, normalizacion_paciente.py, control_sesion.py
Carpetas (cada fichero con una sola responsabilidad):
    modelos/     cada modelo clásico en su fichero.
    evaluacion/  AUROC global y por paciente, sensibilidad, Evaluador y lectura de diferencias (antes metricas.py).
    auditoria/   pistas tramposas de F02.05 (antes atajos.py).
    contextos/   el contexto de una ventana: duración, disponibilidad, cálculo, índices y resumen (antes contexto.py).
"""
