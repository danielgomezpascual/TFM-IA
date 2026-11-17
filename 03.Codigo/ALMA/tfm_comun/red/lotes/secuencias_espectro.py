"""Secuencias de espectros (SecuenciasEspectro): las secuencias de un estudio, listas para servir lotes a la red.

Motivo para cambiar: que cambie cómo se juntan, se normalizan o se colocan en memoria las secuencias.

Imagen: fase_03/img/arquitectura_cnn_gru.png (arriba: de las 15 ventanas al lote que entra en la red)

Por qué no se usa el Dataset de F03.01 (red/ventanas/)
======================================================
F03.01 sirve ventanas sueltas de 4 s, en bruto, desde el paquete en disco. F03.03 decidió que la red recibe
60 s de contexto y a cada paciente comparado consigo mismo. Aquí:
    - cada ventana de 4 s se guarda UNA sola vez (su entrada, red/entradas/), aunque forme parte del
      contexto de muchas filas (dos ⏳ seguidas comparten 14 de sus 15 ventanas);
    - cada fila es solo una lista de 15 índices a esas ventanas (prediccion/contextos/indices.py: indices_contexto);
    - el lote se monta con esos índices directamente en la GPU (si cabe): casi instantáneo, sin DataLoader.

Qué se hace con cada lote, en orden
===================================
1. Juntar las k ventanas de cada fila: (lote, k, 22 canales, 40 frecuencias).
2. Normalizar por paciente con la "calibración tranquila" de F03.03 (prediccion/normalizacion_paciente.py:
   calibracion_por_paciente): a cada número se le resta la mediana de ese paciente y se divide por su escala.
   La MISMA calibración para las k ventanas de la fila: así la red ve cómo evoluciona respecto a lo normal en él.
3. Los NaN (canales sin señal) pasan a 0 = "como lo normal en este paciente", y se recorta a ±recorte.
4. Solo al entrenar: reflejo de hemisferios (red/aumentos/reflejo_hemisferios.py) en una fracción de las filas.
"""

import numpy as np
import torch

from tfm_comun.red.aumentos.reflejo_hemisferios import ORIGEN_ESPEJO


RECORTE = 10.0          # ±10 "escalas" del paciente: más allá es casi seguro un artefacto


class SecuenciasEspectro:
    """Todas las secuencias de un estudio, listas para servir lotes a la red.

    Parámetros:
        valores: (ventanas guardadas, canales x frecuencias) en float32: el espectro relativo de cada ventana de 4 s
            que hace falta (lo que devuelve el almacén).
        indices: (filas, k) int: qué ventanas guardadas forman la secuencia de cada fila (de la más antigua a la actual).
        etiquetas: (filas,) 1 = ⏳ preictal, 0 = 💤 interictal.
        grupo, medianas, escalas: la calibración de cada fila (calibracion_por_paciente).
        n_canales: 22.  dispositivo: dónde se hacen las cuentas (torch.device).
        en_dispositivo: True = los valores se copian a la GPU (en float16: la mitad de memoria). False = se quedan en
            la memoria del ordenador y cada lote se copia a la GPU. 'auto' = a la GPU si caben con holgura.
        recorte: ver RECORTE.
    """

    def __init__(self, valores, indices, etiquetas, grupo, medianas, escalas, n_canales, dispositivo,
                 en_dispositivo='auto', recorte=RECORTE):
        self.dispositivo = torch.device(dispositivo)
        self.n_canales = int(n_canales)
        self.n_frecuencias = valores.shape[1] // self.n_canales
        self.recorte = recorte
        if en_dispositivo == 'auto':
            en_dispositivo = self._cabe_en_gpu(valores)
        self.en_dispositivo = bool(en_dispositivo)
        lugar = self.dispositivo if self.en_dispositivo else torch.device('cpu')
        forma = (len(valores), self.n_canales, self.n_frecuencias)
        self.valores = torch.as_tensor(np.asarray(valores, dtype=np.float16).reshape(forma)).to(lugar)
        self.indices = torch.as_tensor(np.asarray(indices, dtype=np.int64)).to(lugar)
        self.grupo = torch.as_tensor(np.asarray(grupo, dtype=np.int64)).to(lugar)
        self.medianas = torch.as_tensor(np.asarray(medianas, dtype=np.float32).reshape(-1, self.n_canales, self.n_frecuencias)).to(self.dispositivo)
        self.escalas = torch.as_tensor(np.asarray(escalas, dtype=np.float32).reshape(-1, self.n_canales, self.n_frecuencias)).to(self.dispositivo)
        self.etiquetas = torch.as_tensor(np.asarray(etiquetas, dtype=np.float32)).to(self.dispositivo)
        self.k_max = self.indices.shape[1]
        self.origen_espejo = torch.as_tensor(ORIGEN_ESPEJO, device=self.dispositivo)

    def _cabe_en_gpu(self, valores):
        if self.dispositivo.type != 'cuda':
            return self.dispositivo.type == 'mps'
        libre, _ = torch.cuda.mem_get_info(self.dispositivo)
        return libre > 3 * valores.size * 2          # float16 = 2 bytes; holgura para la red y los lotes

    @property
    def gb(self):
        """Memoria que ocupan los espectros guardados (GB)."""
        return self.valores.numel() * 2 / 1e9

    def __len__(self):
        return len(self.indices)

    def lote(self, posiciones, k=None, reflejo=0.0, generador=None):
        """El lote de unas filas: (x, y).

        Parámetros:
            posiciones: filas del lote (array de enteros).  k: cuántas ventanas de la secuencia se usan, contando
                desde la actual hacia atrás (por defecto, todas: 15 = 60 s; 1 = solo la ventana actual, 4 s).
            reflejo: fracción de filas que se reflejan izquierda <-> derecha (solo al entrenar).
            generador: torch.Generator para que el reflejo sea repetible.

        Devuelve:
            x: tensor float32 (lote, k, 22, 40) en el dispositivo.  y: tensor float32 (lote,).
        """
        k = k or self.k_max
        pos = torch.as_tensor(np.asarray(posiciones), dtype=torch.int64)
        pos_datos = pos.to(self.indices.device)
        idx = self.indices[pos_datos][:, self.k_max - k:]                       # (lote, k)
        x = self.valores[idx].to(self.dispositivo, non_blocking=True).float()   # (lote, k, canales, frecuencias)
        g = self.grupo[pos_datos].to(self.dispositivo)
        x = (x - self.medianas[g].unsqueeze(1)) / self.escalas[g].unsqueeze(1)
        x = torch.nan_to_num(x, nan=0.0).clamp_(-self.recorte, self.recorte)
        if reflejo > 0:
            elegidas = torch.rand(len(x), generator=generador) < reflejo
            if elegidas.any():
                elegidas = elegidas.to(self.dispositivo)
                x[elegidas] = x[elegidas][:, :, self.origen_espejo]   # en el espectro el signo del canal no importa
        return x, self.etiquetas[pos.to(self.dispositivo)]
