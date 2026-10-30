"""
Particiones del dataset TUSZ.

Motivo para cambiar: que cambien las particiones del dataset.

TUSZ viene ya repartido POR PACIENTE (ningún paciente está en dos particiones):
    train -> entrenar el modelo.
    dev   -> validar: ajustar hiperparámetros y umbrales.
    eval  -> test final: solo se mira al final, para dar el resultado.
Este orden es el que siguen todas las tablas y figuras del TFM.
"""

SPLITS = ['train', 'dev', 'eval']
