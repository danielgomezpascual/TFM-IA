"""Red CNN + GRU (F03.04): la primera red neuronal de predicción (⏳ preictal frente a 💤 interictal), en PyTorch.

Motivo para cambiar: que cambie la arquitectura de la red CNN + GRU.

Imágenes: fase_03/img/arquitectura_cnn_gru.png (la red completa) y fase_03/img/gru_por_dentro.png (qué hace la GRU)

La red tiene dos piezas
=======================
1. CodificadorVentana (una CNN, red convolucional; en red/redes/codificador_ventana.py): mira el espectro de UNA ventana de 4 s (22 canales x 40
   frecuencias) y lo resume en 64 números. Las convoluciones recorren el eje de la frecuencia con filtros de
   3 Hz: aprenden formas como "pico en 8-12 Hz" o "mucha energía lenta", y combinan los 22 canales.
   Es la MISMA red para las 15 ventanas (comparte los pesos): una ventana se resume igual esté donde esté.
2. Una GRU (Gated Recurrent Unit, red recurrente): lee los 15 resúmenes en orden, del más antiguo al actual,
   y lleva una memoria de 64 números. En cada paso decide con dos "puertas" qué olvida y qué apunta. Al
   terminar, su memoria resume los 60 s; una capa final la convierte en un número: el logit (ver abajo).

Con k = 1 (una sola ventana) la GRU solo da un paso: es la misma red, pero sin contexto. Sirve para medir
cuánto aporta mirar 60 s dentro de la propia red.

Qué devuelve: un LOGIT (un número cualquiera, de -inf a +inf). La probabilidad de ⏳ es sigmoid(logit): 0 -> 0,5;
+3 -> 0,95; -3 -> 0,05. Se entrena con el logit (BCEWithLogitsLoss) porque es más estable numéricamente.

Tamaño pequeño a propósito (unos 50.000 parámetros): hay unos 80 pacientes en 🔵 train y una red grande se
aprendería a esos pacientes de memoria (sobreajuste). El dropout (apagar al azar el 30 % de las conexiones al
entrenar) va en la misma dirección.
"""

from torch import nn

from tfm_comun.red.redes.codificador_ventana import CodificadorVentana


ARQUITECTURA = {'filtros': 32, 'resumen_ventana': 64, 'memoria_gru': 64, 'dropout': 0.3}


class RedCNNGRU(nn.Module):
    """La red completa: CodificadorVentana para cada ventana + GRU sobre la secuencia + capa final.

    Entrada: (lote, k, canales, frecuencias), con k = ventanas de la secuencia (15 para 60 s, 1 para 4 s).
    Salida: (lote,) logits (más alto = más probable que sea ⏳ preictal).
    """

    def __init__(self, n_canales=22, n_frecuencias=40, filtros=32, resumen_ventana=64, memoria_gru=64, dropout=0.3):
        super().__init__()
        self.codificador = CodificadorVentana(n_canales, n_frecuencias, filtros, resumen_ventana, dropout)
        self.gru = nn.GRU(resumen_ventana, memoria_gru, batch_first=True)
        self.final = nn.Sequential(nn.Dropout(dropout), nn.Linear(memoria_gru, 1))

    def resumir_ventanas(self, x):
        """Los resúmenes de cada ventana: (lote, k, resumen_ventana). Útil para mirar qué ha aprendido."""
        lote, k = x.shape[:2]
        return self.codificador(x.reshape(lote * k, *x.shape[2:])).reshape(lote, k, -1)

    def forward(self, x):
        _, memoria = self.gru(self.resumir_ventanas(x))   # memoria: (1, lote, memoria_gru), la del último paso
        return self.final(memoria[-1]).squeeze(-1)
