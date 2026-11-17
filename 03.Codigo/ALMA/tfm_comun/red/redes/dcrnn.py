"""DCRNN: la red de grafos de Tang et al. (ICLR 2022) para la predicción (F03.08).

Motivo para cambiar: que cambie la arquitectura de la red de grafos.

En sencillo
===========
Es la GRU de F03.06, pero en vez de mirar los 22 canales juntos en una capa densa, lleva UNA memoria por canal y
cada canal, en cada segundo, mezcla su memoria con la de sus vecinos del grafo (red/grafo.py). Las multiplicaciones
de la GRU se cambian por una CONVOLUCIÓN DE DIFUSIÓN: "lo mío + lo de mis vecinos + lo de los vecinos de mis
vecinos" (K = 2 pasos), con pesos que se aprenden.

Por qué puede ayudar aquí: los pesos son los MISMOS para los 22 canales (cada canal entra solo con sus 40 Hz).
La red de F03.06 metía los 880 números de golpe en una capa densa y se aprendía a los pacientes; esta tiene
menos margen para memorizar y, además, ve qué canales se sincronizan (el grafo cambia con cada clip).

Imagen: fase_03/img/grafo_dcrnn.png (el grafo de un clip y la difusión a 1 y 2 pasos)

Piezas (una por fichero)
========================
    ConvolucionDifusion   (red/redes/convolucion_difusion.py) [X, P₁X, P₁²X, P₂X, P₂²X] -> capa lineal (la misma para todos los nodos)
    CeldaDCGRU            (red/redes/celda_dcgru.py) una GRU donde cada multiplicación es una ConvolucionDifusion
    RedDCRNN              (este fichero) 2 capas de CeldaDCGRU sobre los 60 segundos -> por nodo, una capa final -> el máximo entre
                          nodos es el logit (como Tang: basta con que una región "se encienda")
"""

from torch import nn

from tfm_comun.red.grafo import K_VECINOS, grafo_correlacion, matrices_de_paso
from tfm_comun.red.redes.celda_dcgru import CeldaDCGRU


# Arquitectura de la red de grafos (F03.08): DCGRU de 2 capas x 64, K = 2, grafo de correlación con 3 vecinos
ARQUITECTURA_DCRNN = {'tipo': 'dcrnn', 'segundos_por_ventana': 4, 'memoria': 64, 'capas': 2, 'k_difusion': 2, 'k_vecinos': 3,
                      'dropout': 0.3}


class RedDCRNN(nn.Module):
    """La red de grafos completa (predicción ⏳/💤).

    Entrada: (lote, k, canales, segundos_por_ventana x frecuencias), como RedGRUSegundos (red/redes/gru_segundos.py).
    Salida: (lote,) logits.
    """

    def __init__(self, n_canales=22, n_frecuencias=160, segundos_por_ventana=4, memoria=64, capas=2, k_difusion=2,
                 k_vecinos=K_VECINOS, dropout=0.3):
        super().__init__()
        assert n_frecuencias % segundos_por_ventana == 0
        self.n_canales, self.segundos = n_canales, segundos_por_ventana
        self.frecuencias = n_frecuencias // segundos_por_ventana
        self.k_vecinos = k_vecinos
        self.celdas = nn.ModuleList([CeldaDCGRU(self.frecuencias if i == 0 else memoria, memoria, k_difusion) for i in range(capas)])
        self.final = nn.Sequential(nn.Dropout(dropout), nn.Linear(memoria, 1))

    def secuencia_de_segundos(self, x):
        """(lote, k, canales, segundos x frecuencias) -> (lote, k x segundos, canales, frecuencias)."""
        lote, k = x.shape[:2]
        x = x.reshape(lote, k, self.n_canales, self.segundos, self.frecuencias).permute(0, 1, 3, 2, 4)
        return x.reshape(lote, k * self.segundos, self.n_canales, self.frecuencias)

    def forward(self, x):
        x = self.secuencia_de_segundos(x)                                # (lote, pasos, nodos, frecuencias)
        matrices = matrices_de_paso(grafo_correlacion(x, self.k_vecinos))
        lote, pasos, nodos, _ = x.shape
        h = [x.new_zeros(lote, nodos, celda.memoria) for celda in self.celdas]
        for t in range(pasos):
            entrada = x[:, t]
            for i, celda in enumerate(self.celdas):
                h[i] = celda(entrada, h[i], matrices)
                entrada = h[i]
        por_nodo = self.final(h[-1]).squeeze(-1)                        # (lote, nodos): cuánto "se enciende" cada canal
        return por_nodo.max(dim=-1).values
