"""
Tipografía de las figuras.

Motivo para cambiar: que cambie la letra de las figuras.
Las figuras del TFM van en Times (tipo LaTeX, como la memoria). Se usa como contexto, para no
cambiar el resto de figuras del notebook:

    with plt.rc_context(ESTILO_TIMES):
        fig, ax = plt.subplots()
        ...
"""

ESTILO_TIMES = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],   # la primera que exista
    'mathtext.fontset': 'stix',                                                  # fórmulas tipo LaTeX
}
