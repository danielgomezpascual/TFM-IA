"""
Panel "¿dónde se origina la crisis?" del visor, con lo que miden los electrodos (sin el cerebro 3D).

Motivo para cambiar: que cambie cómo se enseña el aumento de potencia por electrodo en el visor.
El cerebro 3D (sLORETA) se dibuja con figura_desde_arriba() de cerebro.py, la misma de F01.03v.
Misma escala de color que F01.03v (cerebro.py): gris por debajo del 60 % del máximo y de amarillo a
rojo oscuro por encima, así una crisis focal enciende una zona pequeña y una generalizada casi toda la cabeza.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from tfm_comun.visual.cerebro import MAPA_COLOR, limites_color
from tfm_comun.visual.colores import COLORES_ESQUEMA
from tfm_comun.visual.mapa_cabeza import dibujar_mapa_cabeza
from tfm_comun.visual.tipografia import ESTILO_TIMES

N_BARRAS = 10          # electrodos que se enseñan en las barras (los que más crecen)
MINIMO_CRECE_DB = 1    # si ningún electrodo crece más de 1 dB (x1,26), se considera que no crece ninguno


def figura_aumento_electrodos(aumento_db, titulo, subtitulo):
    """Cabeza con el aumento de potencia por electrodo (izquierda) y los que más crecen (derecha).

    Parámetros:
        aumento_db: pd.Series {electrodo: dB}, de aumento_por_electrodo().
        titulo, subtitulo: cabecera (título en negro con línea roja debajo, como las figuras del TFM).

    Devuelve:
        La figura de matplotlib.
    """
    # Si ningún electrodo crece (p. ej. el basal estaba más activo), la regla del 60 % lo dejaría todo gris:
    # entonces se colorea de menos a más, y el título lo avisa
    ninguno_crece = aumento_db.max() <= MINIMO_CRECE_DB
    limites = ({'fmin': aumento_db.min(), 'fmax': aumento_db.max()} if ninguno_crece
               else limites_color(aumento_db.values))
    with plt.rc_context(ESTILO_TIMES):
        fig, (ax_cabeza, ax_barras) = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=[1, 1.5])
        imagen = dibujar_mapa_cabeza(ax_cabeza, aumento_db.to_dict(), MAPA_COLOR,
                                     vmin=limites['fmin'], vmax=limites['fmax'])
        fig.colorbar(imagen, ax=ax_cabeza, shrink=0.7, extend='neither' if ninguno_crece else 'min',
                     label='aumento de potencia (dB)')
        ax_cabeza.set_title('Ninguno crece: rojo = el que menos baja' if ninguno_crece
                            else 'Cuero cabelludo: dónde crece la potencia', fontsize=11, fontweight='bold',
                            color=COLORES_ESQUEMA['rojo'] if ninguno_crece else COLORES_ESQUEMA['negro'])

        primeros = aumento_db.head(N_BARRAS)[::-1]
        # Cada barra con el mismo color que su electrodo en la cabeza (gris = por debajo del 60 % del máximo)
        colores = MAPA_COLOR(Normalize(limites['fmin'], limites['fmax'])(primeros.values))
        ax_barras.barh(primeros.index, primeros.values, color=colores, edgecolor='black', lw=0.5)
        ax_barras.axvline(0, color=COLORES_ESQUEMA['negro'], lw=0.8)
        if not ninguno_crece:
            ax_barras.axvline(limites['fmin'], color=COLORES_ESQUEMA['rojo'], ls='--', lw=1)
            ax_barras.text(limites['fmin'], 0.01, ' 60 % del máximo',
                           transform=ax_barras.get_xaxis_transform(), color=COLORES_ESQUEMA['rojo'], fontsize=8,
                           va='bottom')
        ax_barras.set_xlabel('aumento de potencia respecto al basal (dB) · +10 dB = 10 veces más')
        ax_barras.set_title('Electrodos que más crecen (mismo color que en la cabeza)', fontsize=11,
                            fontweight='bold')
        for lado in ('top', 'right'):
            ax_barras.spines[lado].set_visible(False)

        fig.suptitle(titulo, x=0.02, y=0.99, ha='left', fontsize=14, fontweight='bold',
                     color=COLORES_ESQUEMA['negro'])
        fig.text(0.02, 0.905, subtitulo, fontsize=10, color=COLORES_ESQUEMA['gris'])
        fig.add_artist(plt.Line2D([0.02, 0.98], [0.94, 0.94], color=COLORES_ESQUEMA['rojo'], lw=1.5))
        fig.subplots_adjust(top=0.8, bottom=0.12, left=0.02, right=0.98, wspace=0.25)
    return fig
