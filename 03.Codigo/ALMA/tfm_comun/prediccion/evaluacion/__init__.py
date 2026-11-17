"""Evaluación: cómo se mide lo bien que separa un modelo (o una variable) las ⏳ preictales de las 💤 interictales.

Imágenes: fase_03/img/auc_roc.png (qué es la AUC-ROC) y fase_03/img/auc_dentro_paciente.png (global frente a
dentro del paciente)

    bootstrap.py      -> cuántas repeticiones y qué nivel de confianza se usan en los intervalos por paciente.
    auroc_global.py   -> la AUROC con todas las ventanas juntas, su intervalo y la diferencia entre dos modelos.
    auroc_paciente.py -> la AUROC de cada paciente y la AUROC DENTRO del paciente (lo que necesita una app).
    sensibilidad.py   -> un punto de la curva que se entiende: sensibilidad con especificidad del 90 %.
    evaluador.py      -> Evaluador: todas las notas de unas puntuaciones en las mismas ventanas, y la comparación
                         emparejada de dos modelos (F03.05-F03.08 lo tenían copiado en cada notebook).
    lectura.py        -> cómo se lee en palabras una diferencia con su intervalo ("mejora real", "empate"...).
"""



