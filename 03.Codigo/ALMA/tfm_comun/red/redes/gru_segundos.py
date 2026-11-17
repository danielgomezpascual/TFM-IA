"""
Red GRU por segundos (F03.06): la red al estilo de Tang et al. (ICLR 2022), sin el grafo.

Motivo para cambiar: que cambie la arquitectura de la red GRU por segundos.

También la usa el preentrenamiento (red/redes/preentreno.py: su proyección y su GRU son el codificador que se reutiliza).
"""

from torch import nn

ARQUITECTURA_SEGUNDOS = {'tipo': 'gru_segundos', 'segundos_por_ventana': 4, 'resumen_paso': 64, 'memoria_gru': 64,
                         'capas_gru': 2, 'dropout': 0.3}

# Red pequeña para la entrada de bandas por segundo (110 números por segundo): unos 10.000 parámetros
ARQUITECTURA_BANDAS = {'tipo': 'gru_segundos', 'segundos_por_ventana': 4, 'resumen_paso': 32, 'memoria_gru': 32,
                       'capas_gru': 1, 'dropout': 0.5}


class RedGRUSegundos(nn.Module):
    """La red al estilo de Tang et al. (ICLR 2022), sin el grafo: una GRU que lee el clip SEGUNDO A SEGUNDO (F03.06).

    Imagen: no tiene todavía (se hará con los resultados).

    Qué recibe: la log-FFT de cada segundo (red/entradas/log_fft_segundos.py: log_fft_por_segundo), guardada por ventanas de 4 s.
    Entrada: (lote, k, canales, segundos_por_ventana x frecuencias), p. ej. (lote, 15, 22, 160) para 60 s.
    Se reordena a la secuencia de segundos: (lote, 60, 22 x 40).

    Piezas:
        1. Proyección (la misma para cada segundo): los 22 x 40 = 880 números de un segundo -> `resumen_paso`
           (64) con una capa lineal + LayerNorm + GELU + dropout. Es el equivalente a la entrada de la GRU de Tang,
           que mira todos los canales y frecuencias de ese segundo a la vez.
        2. GRU de `capas_gru` capas (Tang: 2 capas de 64) sobre los 60 segundos, del más antiguo al actual.
        3. Capa final: la memoria del último paso -> un logit (como RedCNNGRU).

    Diferencias con Tang: su DCRNN cambia las multiplicaciones de la GRU por una difusión sobre el grafo de
    electrodos (cada nodo mira a sus vecinos). Aquí todos los canales entran juntos en la proyección, como su
    variante sin grafo (LSTM, AUROC 0,786 en detección frente a 0,812-0,824 de sus DCRNN sin preentrenar).
    El grafo es el siguiente paso si esta red funciona.
    """

    def __init__(self, n_canales=22, n_frecuencias=160, segundos_por_ventana=4, resumen_paso=64, memoria_gru=64,
                 capas_gru=2, dropout=0.3):
        super().__init__()
        assert n_frecuencias % segundos_por_ventana == 0, 'n_frecuencias tiene que ser segundos_por_ventana x frecuencias por segundo.'
        self.n_canales, self.segundos = n_canales, segundos_por_ventana
        self.frecuencias = n_frecuencias // segundos_por_ventana
        self.proyeccion = nn.Sequential(nn.Linear(n_canales * self.frecuencias, resumen_paso), nn.LayerNorm(resumen_paso),
                                        nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(resumen_paso, memoria_gru, num_layers=capas_gru, batch_first=True,
                          dropout=dropout if capas_gru > 1 else 0.0)
        self.final = nn.Sequential(nn.Dropout(dropout), nn.Linear(memoria_gru, 1))

    def secuencia_de_segundos(self, x):
        """(lote, k, canales, segundos x frecuencias) -> (lote, k x segundos, canales x frecuencias), en orden de tiempo."""
        lote, k = x.shape[:2]
        x = x.reshape(lote, k, self.n_canales, self.segundos, self.frecuencias).permute(0, 1, 3, 2, 4)
        return x.reshape(lote, k * self.segundos, self.n_canales * self.frecuencias)

    def forward(self, x):
        _, memoria = self.gru(self.proyeccion(self.secuencia_de_segundos(x)))   # memoria: (capas, lote, memoria_gru)
        return self.final(memoria[-1]).squeeze(-1)
