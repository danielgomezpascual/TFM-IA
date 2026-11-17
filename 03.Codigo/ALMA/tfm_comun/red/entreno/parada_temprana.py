"""Parada temprana (early stopping): recordar la mejor medida en validación y decidir cuándo dejar de entrenar.

Motivo para cambiar: que cambie el criterio para guardar la mejor red o para parar.

Imagen: fase_03/img/parada_temprana.png (las curvas de aprendizaje y el momento de parar)

La idea
=======
Al entrenar, la nota en validación sube al principio; llega un momento en que la red empieza a aprenderse a los
pacientes de 🔵 train (la pérdida de train sigue bajando) y la nota de 🟡 dev se estanca o baja. ParadaTemprana:
    1. guarda una copia de los pesos cada vez que la nota mejora (al menos `margen`);
    2. cuenta cuántas medidas seguidas lleva sin mejorar;
    3. cuando llega a `paciencia` medidas sin mejorar, avisa de que hay que parar (atributo parar = True).
Al terminar, la red se queda con los pesos de la MEJOR medida, no con los de la última.

Por qué es una clase aparte: entrenar_red solo tiene que llamar a actualizar() después de cada medida. Así el bucle
de entrenamiento no mezcla "cómo se aprende" con "cuándo se para", y la regla se puede cambiar sin tocar el bucle.
"""

import copy

import numpy as np


class ParadaTemprana:
    """Recuerda la mejor medida de validación y avisa cuando lleva `paciencia` medidas sin mejorar.

    Parámetros:
        paciencia: medidas seguidas sin mejorar que se aguantan antes de parar (épocas x medidas por época).
        margen: lo mínimo que tiene que subir la nota para contar como mejora (evita "mejoras" de ruido).

    Atributos (lo que se guarda en el punto de control):
        mejor_nota, mejor_estado (copia de los pesos), mejor_momento (época, puede ser fraccionaria),
        sin_mejorar (medidas seguidas sin mejorar), parar (True cuando se agota la paciencia).
    """

    def __init__(self, paciencia, margen=1e-4):
        self.paciencia, self.margen = paciencia, margen
        self.mejor_nota, self.mejor_estado, self.mejor_momento = -np.inf, None, 0
        self.sin_mejorar, self.parar = 0, False

    def actualizar(self, nota, modelo, momento):
        """Apunta una medida. Devuelve True si es la mejor hasta ahora (y entonces copia los pesos del modelo)."""
        if nota > self.mejor_nota + self.margen:
            self.mejor_nota, self.mejor_estado, self.mejor_momento = nota, copy.deepcopy(modelo.state_dict()), momento
            self.sin_mejorar = 0
            return True
        self.sin_mejorar += 1
        if self.sin_mejorar >= self.paciencia:
            self.parar = True
        return False

    def estado(self):
        """Lo que hay que guardar en el punto de control (mismas claves que antes del cambio: los .pt antiguos valen)."""
        return {'mejor_auc': self.mejor_nota, 'mejor_estado': self.mejor_estado, 'mejor_epoca': self.mejor_momento,
                'sin_mejorar': self.sin_mejorar, 'parar': self.parar}

    def cargar(self, punto):
        """Recupera el estado guardado con estado() (al seguir un entrenamiento cortado)."""
        self.mejor_nota, self.mejor_estado = punto['mejor_auc'], punto['mejor_estado']
        self.mejor_momento, self.sin_mejorar, self.parar = punto['mejor_epoca'], punto['sin_mejorar'], punto['parar']
