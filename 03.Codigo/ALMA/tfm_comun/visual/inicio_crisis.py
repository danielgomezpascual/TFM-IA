"""
Figura del inicio de la crisis, al estilo de las de SEEG: qué canales arrancan primero y cómo.

Motivo para cambiar: que cambie cómo se enseña el arranque de la crisis canal a canal.
En SEEG (electrodos dentro del cerebro) la figura clásica pone las trazas de los contactos alrededor
del inicio, resalta los que arrancan antes y debajo el espectrograma del contacto principal
[Bartolomei et al., 2008]. Aquí se hace lo mismo con los canales TCP de superficie de TUSZ, y se
añade la cabeza con los canales que más crecen.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from tfm_comun.senal.montaje import PARES_TCP
from tfm_comun.visual.cabeza import dibujar_contorno, dibujar_electrodos, dibujar_pares
from tfm_comun.visual.tipografia import ESTILO_TIMES

ROJO_OSCURO = '#8b0000'


def aumento_por_canal(senal_uv, tiempos, canales, tramo_basal, tramo_inicio):
    """Aumento de potencia (dB) de cada canal en el tramo de inicio respecto al basal.

    +10 dB = 10 veces más potencia; 0 dB = sin cambio. Devuelve un dict {canal: dB}.
    """
    def potencia(tramo):
        dentro = (tiempos >= tramo[0]) & (tiempos < tramo[1])
        return (senal_uv[:, dentro] ** 2).mean(axis=1)
    return dict(zip(canales, 10 * np.log10(potencia(tramo_inicio) / potencia(tramo_basal))))


def figura_inicio_crisis(senal_uv, tiempos, fs, canales, aumento_db, canales_lider, t_crisis, t_fin_crisis,
                         color_crisis, titulo, subtitulo, antes_s=10, despues_s=15, n_canales=10):
    """Figura del arranque de la crisis: cabeza, trazas de los canales que más crecen y espectrograma.

    Parámetros:
        senal_uv: (canales, muestras) en µV, ya filtrada. tiempos: segundos de cada muestra. fs: Hz.
        canales: nombre de cada fila de senal_uv ('FP1-F7'...).
        aumento_db: {canal: aumento en dB} (de aumento_por_canal).
        canales_lider: los canales que arrancan con más fuerza (se pintan en rojo).
        t_crisis, t_fin_crisis: inicio y fin anotados de la crisis (s).
        color_crisis: color del tipo de crisis (sombreado). antes_s, despues_s: ventana alrededor del inicio.
        n_canales: cuántos canales se dibujan (los que más crecen).

    Paneles:
        Izquierda: cabeza con los canales TCP (flechas); en rojo los líderes.
        Derecha arriba: trazas de los n canales que más crecen, ordenados de más a menos.
        Derecha abajo: espectrograma del canal líder (cómo cambian sus frecuencias al arrancar).
    """
    orden = sorted(aumento_db, key=aumento_db.get, reverse=True)[:n_canales]
    t_ini, t_fin = t_crisis - antes_s, min(t_crisis + despues_s, tiempos[-1])
    tramo = (tiempos >= t_ini) & (tiempos <= t_fin)
    t_rel = tiempos[tramo] - t_crisis

    with plt.rc_context(ESTILO_TIMES):
        fig = plt.figure(figsize=(15, 10))
        rejilla = fig.add_gridspec(2, 2, width_ratios=[1, 2.6], height_ratios=[2.2, 1], wspace=0.08, hspace=0.12)
        ax_cabeza = fig.add_subplot(rejilla[:, 0])
        ax_trazas = fig.add_subplot(rejilla[0, 1])
        ax_espectro = fig.add_subplot(rejilla[1, 1], sharex=ax_trazas)

        # Cabeza: canales TCP en gris y los líderes en rojo
        pares = [p for p in PARES_TCP if f'{p[0]}-{p[1]}' in canales]
        lider = [p for p in pares if f'{p[0]}-{p[1]}' in canales_lider]
        dibujar_contorno(ax_cabeza)
        dibujar_pares(ax_cabeza, [p for p in pares if p not in lider], color='lightgray', ancho=1.2)
        dibujar_pares(ax_cabeza, lider, color=ROJO_OSCURO, ancho=3.2)
        dibujar_electrodos(ax_cabeza, sorted({e for p in pares for e in p}), tamano=0.075, fuente=7,
                           color='white')
        for texto in ax_cabeza.texts:
            texto.set_color('black')
        ax_cabeza.set_title('Canales que arrancan con más fuerza', fontsize=13, fontweight='bold')
        ax_cabeza.text(0, -1.22, '\n'.join(f'{c}: +{aumento_db[c]:.0f} dB'.replace('.', ',') for c in canales_lider),
                       ha='center', va='top', fontsize=11, color=ROJO_OSCURO, fontweight='bold')

        # Trazas: carril de cada canal = 3 veces su RMS en el tramo, recortado como en los visores clínicos
        indices = [canales.index(c) for c in orden]
        separacion = 3 * np.sqrt((senal_uv[indices][:, tramo] ** 2).mean(axis=1)).max()
        for fila, (canal, i) in enumerate(zip(orden, indices)):
            color = ROJO_OSCURO if canal in canales_lider else 'black'
            traza = np.clip(senal_uv[i, tramo], -separacion, separacion)
            ax_trazas.plot(t_rel, traza - fila * separacion, color=color, linewidth=0.7)
        ax_trazas.set_yticks([-f * separacion for f in range(len(orden))])
        ax_trazas.set_yticklabels([f'{c}  (+{aumento_db[c]:.0f} dB)'.replace('.', ',') for c in orden], fontsize=9)
        for etiqueta, canal in zip(ax_trazas.get_yticklabels(), orden):
            etiqueta.set_color(ROJO_OSCURO if canal in canales_lider else 'black')
        ax_trazas.set_ylim(-(len(orden) - 0.4) * separacion, separacion * 1.1)
        ax_trazas.set_title(f'Los {len(orden)} canales que más crecen al inicio (de más a menos)', fontsize=13,
                            fontweight='bold')
        ax_trazas.tick_params(labelbottom=False)

        # Espectrograma del canal líder: ventanas de 1 s cada 0,1 s
        i_lider = canales.index(orden[0])
        frecs, t_esp, potencia = signal.spectrogram(senal_uv[i_lider, tramo], fs=fs, nperseg=int(fs),
                                                    noverlap=int(0.9 * fs))
        visibles = (frecs >= 0.5) & (frecs <= 30)
        potencia_db = 10 * np.log10(potencia[visibles] + 1e-12)
        ax_espectro.pcolormesh(t_esp + t_rel[0], frecs[visibles], potencia_db, shading='auto', cmap='magma',
                               vmin=np.percentile(potencia_db, 5), vmax=np.percentile(potencia_db, 99.5))
        ax_espectro.set_ylabel('Frecuencia (Hz)')
        ax_espectro.set_xlabel('Tiempo respecto al inicio anotado de la crisis (s)')
        ax_espectro.set_title(f'Espectrograma de {orden[0]}', fontsize=12, fontweight='bold', loc='left')

        # Inicio anotado (línea roja) y duración de la crisis (sombreado)
        for ax in (ax_trazas, ax_espectro):
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax_trazas.axvspan(0, t_fin_crisis - t_crisis, color=color_crisis, alpha=0.12, zorder=0)
        ax_trazas.text(0.2, separacion * 0.95, 'inicio anotado', color='red', fontsize=10, va='bottom')
        ax_trazas.set_xlim(t_rel[0], t_rel[-1])
        for ax in (ax_trazas, ax_espectro):
            ax.spines[['top', 'right']].set_visible(False)

        fig.suptitle(titulo, fontsize=17, fontweight='bold', y=0.995)
        fig.text(0.5, 0.947, subtitulo, ha='center', fontsize=11.5, color='dimgray')
        fig.add_artist(plt.Line2D([0.06, 0.94], [0.937, 0.937], color=ROJO_OSCURO, linewidth=2,
                                  transform=fig.transFigure))
        fig.subplots_adjust(top=0.9, bottom=0.06, left=0.03, right=0.98)
    return fig
