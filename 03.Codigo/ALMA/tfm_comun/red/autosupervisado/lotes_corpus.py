"""Lotes del corpus (F03.07): de unas posiciones de inicio a (contexto, futuro, máscara), normalizados por paciente.

Motivo para cambiar: que cambie cómo se arma un lote del preentrenamiento.

Imagen: fase_03/img/preentreno_autosupervisado.png (contexto de 60 s -> futuro de 12 s)

Cada ejemplo son k_contexto + k_futuro ventanas de 4 s seguidas del corpus:
    x        = las k_contexto primeras (15 = 60 s): lo que ve la red;
    objetivo = las k_futuro siguientes (3 = 12 s), reordenadas en segundos: lo que tiene que predecir;
    máscara  = True donde el objetivo existe (los canales sin señal no cuentan en la pérdida).
Se normaliza como en el ajuste fino (red/lotes/secuencias_espectro.py): mediana y escala del paciente, recorte a ±10
y los NaN a 0.
"""

import numpy as np
import torch


RECORTE = 10.0


class LotesCorpus:
    """Saca lotes de (contexto, futuro) del corpus, ya normalizados por paciente, en la GPU si cabe."""

    def __init__(self, corpus, indice_paciente, medianas, escalas, dispositivo, k_contexto, k_futuro):
        self.dispositivo = torch.device(dispositivo)
        self.k_contexto, self.k_futuro = k_contexto, k_futuro
        libre = torch.cuda.mem_get_info(self.dispositivo)[0] if self.dispositivo.type == 'cuda' else 0
        self.en_gpu = libre > 3 * corpus.valores.nbytes
        lugar = self.dispositivo if self.en_gpu else torch.device('cpu')
        self.valores = torch.as_tensor(corpus.valores).to(lugar)
        self.indice_paciente = torch.as_tensor(indice_paciente).to(lugar)
        self.medianas = torch.as_tensor(medianas).to(self.dispositivo)
        self.escalas = torch.as_tensor(escalas).to(self.dispositivo)
        self.n_canales = 22

    def lote(self, inicios):
        """(x, objetivo, máscara): x (lote, k_contexto, 22, 160); objetivo y máscara (lote, k_futuro x 4, 22 x 40)."""
        k = self.k_contexto + self.k_futuro
        idx = torch.as_tensor(np.asarray(inicios)[:, None] + np.arange(k)[None, :]).to(self.valores.device)
        v = self.valores[idx].to(self.dispositivo, non_blocking=True).float()                       # (lote, k, 3520)
        p = self.indice_paciente[idx[:, 0]].to(self.dispositivo)
        existe = ~torch.isnan(v)
        v = ((v - self.medianas[p].unsqueeze(1)) / self.escalas[p].unsqueeze(1))
        v = torch.nan_to_num(v, nan=0.0).clamp_(-RECORTE, RECORTE)
        lote = v.shape[0]
        v = v.reshape(lote, k, self.n_canales, -1)
        existe = existe.reshape(lote, k, self.n_canales, -1)
        x, futuro, m = v[:, :self.k_contexto], v[:, self.k_contexto:], existe[:, self.k_contexto:]
        return x, self._en_segundos(futuro), self._en_segundos(m)

    def _en_segundos(self, t):
        """(lote, k, 22, 4 x 40) -> (lote, k x 4, 22 x 40), en orden de tiempo (como RedGRUSegundos)."""
        lote, k, c, n = t.shape
        return t.reshape(lote, k, c, 4, n // 4).permute(0, 1, 3, 2, 4).reshape(lote, k * 4, c * (n // 4))
