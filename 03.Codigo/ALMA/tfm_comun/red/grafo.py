"""
Grafo de electrodos: qué canales están "conectados" en cada clip y cómo se reparte la información entre ellos.

Motivo para cambiar: que cambie cómo se construye el grafo (correlación, distancia...) o cómo se normaliza.

El grafo de correlación (Tang et al., ICLR 2022, §3.2)
======================================================
Cada canal TCP es un NODO. Dos nodos están unidos si su actividad en el clip se parece (correlación alta):
    1. Para cada canal, todos sus números del clip (segundos x frecuencias) en fila.
    2. Correlación de Pearson entre cada pareja de canales, en valor absoluto (0 = nada que ver, 1 = igual).
    3. Cada nodo se queda solo con sus K_VECINOS vecinos más parecidos (Tang: 3) y consigo mismo. El grafo
       resultante es DIRIGIDO: que A elija a B no obliga a B a elegir a A.
Un grafo por clip: cambia con lo que pasa en esos 60 s (en una crisis, regiones enteras se sincronizan).
No necesita la posición de los electrodos, así que sirve tal cual con el montaje TCP.

Las "matrices de paso" (doble paseo aleatorio, DCRNN de Li et al., 2018)
========================================================================
La DCGRU (red/redes/dcrnn.py) reparte la información por el grafo con matrices de paso:
    - hacia delante: D_salida⁻¹ · A   (cada nodo promedia a los vecinos que él eligió)
    - hacia atrás:  D_entrada⁻¹ · Aᵀ  (cada nodo promedia a los que le eligieron a él)
Con K pasos de difusión, un nodo "oye" a los vecinos de sus vecinos (K = 2).
"""

import torch

K_VECINOS = 3


def grafo_correlacion(x, k_vecinos=K_VECINOS):
    """La matriz de adyacencia de cada clip: |correlación| entre canales, con solo los k vecinos más fuertes.

    Parámetros:
        x: tensor (lote, pasos, canales, frecuencias): el clip ya normalizado (red/lotes/secuencias_espectro.py).

    Devuelve:
        Tensor (lote, canales, canales): fila i = a quién escucha el canal i (pesos de 0 a 1; 1 en la diagonal).
    """
    lote, pasos, canales, frecuencias = x.shape
    v = x.permute(0, 2, 1, 3).reshape(lote, canales, pasos * frecuencias)
    v = v - v.mean(dim=-1, keepdim=True)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)            # un canal sin señal (todo 0) queda sin vecinos fuertes
    correlacion = torch.bmm(v, v.transpose(1, 2)).abs()
    ojo = torch.eye(canales, device=x.device, dtype=torch.bool).unsqueeze(0)
    sin_diagonal = correlacion.masked_fill(ojo, -1.0)
    umbral = sin_diagonal.topk(k_vecinos, dim=-1).values[..., -1:]    # el k-ésimo más fuerte de cada fila
    adyacencia = torch.where(sin_diagonal >= umbral, correlacion, torch.zeros_like(correlacion))
    return adyacencia.masked_fill(ojo, 1.0)


def matrices_de_paso(adyacencia):
    """Las dos matrices del doble paseo aleatorio: [D_salida⁻¹·A, D_entrada⁻¹·Aᵀ], cada una (lote, canales, canales)."""
    adelante = adyacencia / adyacencia.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    traspuesta = adyacencia.transpose(1, 2)
    atras = traspuesta / traspuesta.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return [adelante, atras]
