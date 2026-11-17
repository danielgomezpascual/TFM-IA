"""Sesiones: a qué sesión (día de grabación) pertenece cada ventana y si esa sesión tiene alguna crisis.

Motivo para cambiar: que cambie cómo se identifica una sesión o qué se considera "sesión con crisis".

Imagen: fase_02/img/misma_sesion.png
"""


def marcar_sesiones(ventanas, edfs):
    """Añade a cada ventana su sesión y si es de una sesión con crisis, para el control de "tipo de fichero".

    Por qué: el 💤 interictal sale casi todo de ficheros sin crisis y el ⏳ preictal de ficheros con crisis. Para
    saber si la red aprende "día con crisis", hace falta 💤 de un día con crisis pero lejos de ellas. El de
    ficheros con crisis ('lejos_de_crisis') casi siempre está JUSTO DESPUÉS de la crisis (EEG aún alterado), así
    que se usa otro: 💤 de ficheros SIN crisis de la MISMA SESIÓN (mismo día) que sí tiene crisis.
    Columnas que añade:
        sesion                     'aaaaaacz_s006'
        sesion_con_crisis          True si algún fichero de la sesión tiene crisis (edfs.n_seiz_events_bi > 0)
        antes_de_la_primera_crisis True si el fichero es un token ANTERIOR al primer fichero con crisis de su sesión
                                   (variante estricta: no puede ser post-crisis; ojo, no se sabe si los tokens
                                   son seguidos ni cuánto tiempo hay entre ellos)
    """
    e = edfs[['file', 'n_seiz_events_bi']].copy()
    e['sesion'] = e.file.str.rsplit('_', n=1).str[0]
    e['token'] = e.file.str.rsplit('_', n=1).str[1].str[1:].astype(int)
    primera = e[e.n_seiz_events_bi > 0].groupby('sesion').token.min()
    v = ventanas.copy()
    v['sesion'] = v.file.str.rsplit('_', n=1).str[0]
    token = v.file.str.rsplit('_', n=1).str[1].str[1:].astype(int)
    v['sesion_con_crisis'] = v.sesion.isin(primera.index)
    v['antes_de_la_primera_crisis'] = v.sesion_con_crisis & (token < v.sesion.map(primera))
    return v
