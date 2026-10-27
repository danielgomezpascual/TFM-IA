"""
Esquemas explicativos (no a escala) de los periodos alrededor de una crisis.

Motivo para cambiar: que cambie cómo se dibuja el esquema de periodos.
Lo usan los notebooks que explican el preictal, el SPH, el lead-in y el margen postictal
(fase_01/03v y fase_02/01), para que todos muestren el mismo dibujo.
"""

import matplotlib.pyplot as plt

from tfm_comun.colores import COLORES_PERIODO, COLOR_DESTACADO
from tfm_comun.numeros import duracion_legible


def _periodo(ax, x0, x1, relleno, texto, color_texto='black', negrita=False, borde='white', rayado=None):
    """Un periodo del esquema: rectángulo de color con su nombre dentro."""
    ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 1, facecolor=relleno, edgecolor=borde, linewidth=2 if rayado is None else 1,
                               hatch=rayado))
    ax.text((x0 + x1) / 2, 0.5, texto, ha='center', va='center', fontsize=9.5, color=color_texto,
            fontweight='bold' if negrita else 'normal')


def _llave(ax, x0, x1, y, texto, color='black'):
    """Corchete horizontal con su texto debajo."""
    ax.plot([x0, x0, x1, x1], [y + 0.12, y, y, y + 0.12], color=color, linewidth=1.3)
    ax.text((x0 + x1) / 2, y - 0.08, texto, ha='center', va='top', fontsize=9.5, color=color)


TEXTO_RESALTADO = {
    'preictal': 'Tramo preictal que se guarda (el más largo): la ventana cian se recorta después',
    'interictal': 'Tramo interictal: señal normal, lejos de cualquier crisis',
    'ictal': 'Tramos ictales: cada crisis, de su inicio a su fin',
}


def dibujar_esquema_periodos(ax, umbral_leadin_s, sph_s, margen_postictal_s, color_crisis='black', resaltar=None):
    """Esquema de los periodos alrededor de una crisis con los parámetros de un tipo (en segundos):
    interictal, crisis anterior, margen postictal, preictal, SPH, crisis y postictal, con las llaves
    del lead-in, del lead-in útil y del umbral. No está a escala.
    resaltar: 'preictal', 'interictal' o 'ictal' para destacar en dorado el tramo que se selecciona
    (el resto del esquema se aclara)."""
    preictal_s = umbral_leadin_s - sph_s
    hay_postictal = margen_postictal_s > 0
    ancho_post = 1.5 if hay_postictal else 0
    texto_post = f'Postictal\n(no se usa:\n{duracion_legible(margen_postictal_s)})'

    # Posiciones de cada periodo (esquema, no a escala)
    fin_anterior = 2.8
    inicio_util = fin_anterior + ancho_post
    inicio_preictal, inicio_sph, inicio_crisis = 6.4, 9.9, 11.0
    fin_crisis = 12.0

    _periodo(ax, 0, 2.0, COLORES_PERIODO['interictal'], 'Interictal\n(clase "normal")', 'white')
    _periodo(ax, 2.0, fin_anterior, color_crisis, 'Crisis\nanterior', 'white', negrita=True)
    if hay_postictal:
        _periodo(ax, fin_anterior, inicio_util, COLORES_PERIODO['postictal'], texto_post)
    _periodo(ax, inicio_util, inicio_preictal, 'white', 'Fuera de la\nventana\n(ni normal\nni preictal)', 'dimgray',
             borde='lightgray', rayado='//')
    _periodo(ax, inicio_preictal, inicio_sph, COLORES_PERIODO['preictal'],
             f'Preictal ({duracion_legible(preictal_s)})\n(clase "se acerca\nuna crisis")')
    _periodo(ax, inicio_sph, inicio_crisis, COLORES_PERIODO['margen'], f'SPH\n({duracion_legible(sph_s)})')
    _periodo(ax, inicio_crisis, fin_crisis, color_crisis, 'Crisis\n(ictal)', 'white', negrita=True)
    if hay_postictal:
        _periodo(ax, fin_crisis, fin_crisis + ancho_post, COLORES_PERIODO['postictal'], texto_post)
    else:
        ax.text(fin_crisis + 0.1, 0.5, 'sin margen\npostictal', va='center', fontsize=9, color='dimgray')

    # Alarma y tiempo para reaccionar
    ax.annotate('Alarma: el modelo avisa', xy=(inicio_sph, 1.02), xytext=(inicio_sph - 1.6, 1.75),
                fontsize=10, color='tab:red', ha='center',
                arrowprops=dict(arrowstyle='-|>', color='tab:red', linewidth=1.5))
    ax.annotate('', xy=(inicio_crisis, 1.25), xytext=(inicio_sph, 1.25),
                arrowprops=dict(arrowstyle='<|-|>', color='tab:red'))
    ax.text((inicio_sph + inicio_crisis) / 2, 1.35, 'tiempo para\nreaccionar', ha='center', va='bottom',
            fontsize=9, color='tab:red')

    # Llaves: lead-in, lead-in útil y umbral
    _llave(ax, fin_anterior, inicio_crisis, -0.25, 'Lead-in: tiempo sin crisis desde la anterior')
    if hay_postictal:
        _llave(ax, inicio_util, inicio_crisis, -0.9,
               f'Lead-in útil = lead-in − margen postictal (≥ {duracion_legible(umbral_leadin_s)} para usar la crisis)')
    else:
        _llave(ax, inicio_util, inicio_crisis, -0.9,
               f'Lead-in útil = lead-in, sin margen postictal (≥ {duracion_legible(umbral_leadin_s)} para usar la crisis)')
    _llave(ax, inicio_preictal, inicio_crisis, -1.55,
           f'Umbral {duracion_legible(umbral_leadin_s)} = {duracion_legible(preictal_s)} de preictal + '
           f'{duracion_legible(sph_s)} de SPH', color='dimgray')

    if resaltar:
        _resaltar(ax, resaltar, {'preictal': [(inicio_util, inicio_sph)], 'interictal': [(0, 2.0)],
                                 'ictal': [(2.0, fin_anterior), (inicio_crisis, fin_crisis)]}[resaltar],
                  fin_crisis + ancho_post)

    ax.annotate('', xy=(14.1, -2.3), xytext=(-0.1, -2.3), arrowprops=dict(arrowstyle='-|>', color='gray'))
    ax.text(14.1, -2.42, 'tiempo', ha='right', va='top', fontsize=9, color='gray')
    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(-2.7, 2.7 if resaltar else 2.2)
    ax.axis('off')


def _resaltar(ax, clase, tramos, fin_barra):
    """Aclara todo el esquema salvo los tramos indicados, que se recuadran en dorado con su explicación."""
    cursor = 0
    for x0, x1 in sorted(tramos) + [(fin_barra, fin_barra)]:
        if x0 > cursor:
            ax.add_patch(plt.Rectangle((cursor, 0), x0 - cursor, 1, facecolor='white', alpha=0.72, edgecolor='none', zorder=3))
        cursor = max(cursor, x1)
    for x0, x1 in tramos:
        ax.add_patch(plt.Rectangle((x0, -0.06), x1 - x0, 1.12, fill=False, edgecolor=COLOR_DESTACADO, linewidth=3.5, zorder=4))
    x_texto = tramos[0][0]
    ax.annotate(f'▼ {TEXTO_RESALTADO[clase]}', xy=(x_texto, 2.35), fontsize=11, fontweight='bold', color=COLOR_DESTACADO,
                ha='left', va='center')
