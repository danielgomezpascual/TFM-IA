# Imágenes de `tfm_comun`

Figuras **sintéticas** (señales inventadas, no datos de TUSZ) que explican qué hace cada función.
Cada función que tiene imagen lo indica en su docstring con una línea `Imagen: tfm_comun/imagenes/<nombre>.png`.
Este índice las muestra todas juntas (se ve bien en GitHub o con la vista previa de Markdown de PyCharm).

## senal/ · transformar la señal

### `montaje.py` · `tcp_desde_referencial()`, `senal_tcp()`, `montaje_tcp()`
Las 5 cadenas del montaje TCP y cómo la resta de dos electrodos cancela la referencia.
![montaje](montaje_tcp_desde_referencial.png)

### `remuestreo.py` · `remuestrear()`
La misma onda medida a 250 Hz y a 256 Hz.
![remuestreo](remuestreo_remuestrear.png)

### `filtrado.py` · por qué este filtro
Butterworth frente a Chebyshev y elíptico, el orden (la "cuesta"), el notch y causal frente a fase cero. Es la única figura con **señal real** (la ausencia de ejemplo, `aaaaacrb_s001_t001`), generada en `fase_02/02_preprocesado.ipynb`.
![explicacion](filtrado_explicacion.png)

### `filtrado.py` · `disenar_filtro()`
Qué porcentaje de cada frecuencia deja pasar el filtro (paso banda 0,5–40 Hz + notch 60 Hz).
![disenar_filtro](filtrado_disenar_filtro.png)

### `filtrado.py` · `filtrar()`
Una señal con deriva, red eléctrica y músculo, antes y después del filtro.
![filtrar](filtrado_filtrar.png)

### `filtrado.py` · `_estado_inicial()`
Sin estado inicial aparece un salto falso al arrancar; con él, el filtro arranca suave.
![estado_inicial](filtrado_estado_inicial.png)

### `filtrado.py` · `FiltroContinuo`
Filtrar de golpe o a trozos (como en la demo en directo) da exactamente lo mismo.
![filtro_continuo](filtrado_filtro_continuo.png)

### `preprocesado.py` · `preprocesar()`, `preprocesar_raw()`, `preprocesar_referencial()`
La cadena completa paso a paso: montaje TCP → µV → 256 Hz → filtro.
![preprocesado](preprocesado_cadena.png)

### `normalizacion.py` · `zscore_por_canal()`, `robusta_por_canal()`
Dos canales de amplitud muy distinta, uno con un artefacto, antes y después de normalizar.
![normalizacion](normalizacion_zscore_robusta.png)

### `calidad.py` · `indicadores_calidad()`
Un canal normal, uno plano, uno con artefacto y uno no válido, con los indicadores que salen.
![calidad](calidad_indicadores.png)

### `senales.py` · `guardar_tramo()`
float32 frente a float16: misma señal, error de centésimas de µV.
![senales](senales_float16.png)

### `bandas.py` · `BANDAS_HZ`
Las bandas clásicas del EEG y una onda de ejemplo de cada una.
![bandas](bandas_eeg.png)

## prediccion/

### `umbrales.py` · `margen_postictal_s()`, `umbral_leadin_s()`, `sph_s()`
Los periodos alrededor de una crisis con los valores por defecto y con los de las ausencias.
![umbrales](umbrales_periodos.png)

### `ventanas.py` · `inicios_ventanas()`, `dentro_del_preictal()`
Cómo se cortan las ventanas de 4 s antes de una crisis y a qué modelo pertenece cada una (señal real: la ausencia de ejemplo, generada en `fase_02/03_ventanas.ipynb`).
![ventanas](ventanas_corte.png)

## dataset/

### `anotaciones.py` · `crisis_del_fichero()`
Inicio y fin del `.csv_bi`; tipo = el que más canales × segundos ocupa en el `.csv`.
![anotaciones](anotaciones_crisis_del_fichero.png)

### `canales.py` · `region_electrodo()`, `lado_electrodo()`
Región y lado de cada electrodo del sistema 10-20.
![canales](canales_region_lado.png)

### `tipos_crisis.py` · `grupo()`
Qué tipos de crisis van en cada grupo de inicio (ILAE).
![tipos](tipos_crisis_grupo.png)

## visual/

### `esquemas.py` · `dibujar_esquema_periodos(..., resaltar=...)`
El esquema de periodos con cada tramo destacado en dorado.
![esquemas](esquemas_resaltar.png)

### `cabeza.py` · `dibujar_contorno()`, `dibujar_electrodos()`, `dibujar_pares()`
La cabeza se dibuja por capas.
![cabeza](cabeza_capas.png)

### `ejes.py` · `eje_log_legible()`
Un eje logarítmico con potencias de 10 y con números normales.
![ejes](ejes_eje_log_legible.png)

### `tablas.py` · `con_separadores()`
![con_separadores](tablas_con_separadores.png)

### `tablas.py` · `mostrar_destacado()`
![mostrar_destacado](tablas_mostrar_destacado.png)
