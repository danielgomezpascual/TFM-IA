# Curso de MNE-Python aplicado a TUSZ

Curso práctico de [MNE-Python](https://mne.tools) pensado para el TFM: de un EDF de TUSZ a un
dataset `X, y, groups` listo para redes neuronales. Todo se trabaja con datos reales de
`D:\Dani\TFM IA\00.TUSZ\v2.0.6`, y el ejemplo principal es la **crisis de ausencia** del paciente `aaaaabqv`.

## Puesta en marcha

1. Instala las librerías (en el entorno del TFM):
   ```bash
   pip install -U mne mne-qt-browser matplotlib pandas scipy scikit-learn jupyter
   ```
   Probado con MNE 1.13; necesitas como mínimo la 1.8.
2. Esta carpeta vive en `D:\Dani\TFM IA\03.Codigo\Curso Claude MNE`. Los notebooks importan `tusz_utils.py`,
   que tiene que estar a su lado.
3. Abre Jupyter en esta carpeta y sigue los notebooks en orden. Lo que generan se guarda en
   `D:\Dani\TFM IA\00.TUSZ\derivados\curso_MNE`.

Rutas: todas se definen en `tusz_utils.py` relativas a la carpeta del proyecto
(`00.TUSZ\v2.0.6`, `02.Notas\tusz_inventory_out` y `00.TUSZ\derivados\curso_MNE`).
Si hace falta, se pueden forzar con las variables de entorno `TUSZ_ROOT`, `TUSZ_INVENTORY` y `TFM_SALIDAS`.

## Temario

| # | Notebook | Qué aprendes | Duración aprox. |
|---|---|---|---|
| 00 | Bienvenida y entorno | Instalación, mapa de objetos de MNE y las tres reglas de oro | 20 min |
| 01 | Raw e Info | Leer un EDF, `info`, limpiar nombres y tipos de canal, `pick`, `crop`, `get_data`, guardar en FIF | 1 h |
| 02 | Visualización | `raw.plot`, PSD, montaje 10-20, mapas topográficos y la punta-onda a 3 Hz | 1 h |
| 03 | Anotaciones TUSZ | `.csv`/`.csv_bi` → `mne.Annotations`, `BAD_`, eventos y máscara de crisis | 1 h |
| 04 | Montajes y referencias | LE vs. AR, montaje bipolar **TCP** de 22 canales y anotaciones por canal | 1 h |
| 05 | Filtrado y remuestreo | Notch a 60 Hz, paso banda FIR/IIR, unificar `sfreq` y la función `preprocess` | 1 h |
| 06 | Artefactos e ICA | Canales malos, interpolación, anotaciones automáticas e ICA (y por qué puede borrar crisis) | 1,5 h |
| 07 | Epochs | Ventanas etiquetadas con `metadata`: ictal, preictal, SPH/SOP, rechazo y balanceo | 1,5 h |
| 08 | Espectro y tiempo-frecuencia | PSD por ventana, potencia por bandas, Morlet y un espectrograma con las crisis encima | 1,5 h |
| 09 | De MNE a ML | Pipeline multi-fichero con caché, `GroupKFold` por paciente, un baseline con sklearn y exportación a PyTorch | 2 h |

Cada módulo trae ejercicios 🧪 con la solución justo debajo.

`tusz_utils.py` reúne las funciones que se van construyendo durante el curso (`read_tusz_raw`,
`tusz_annotations`, `make_tcp`, `preprocess`, `make_windows`, `band_power`…) para que las reutilices
en el TFM.

## Chuleta de MNE

```python
import mne
raw = mne.io.read_raw_edf(path, preload=True)        # leer
raw.info["sfreq"], raw.ch_names, raw.times[-1]        # metadatos
raw.rename_channels(func_o_dict); raw.set_channel_types({...})
raw.pick("eeg"); raw.drop_channels([...]); raw.copy().crop(tmin, tmax)
X = raw.get_data(picks="eeg", tmin=0, tmax=10, units="uV")   # (canales, muestras)

raw.set_annotations(mne.Annotations(onset, duration, description))
events, event_id = mne.events_from_annotations(raw, chunk_duration=None)

raw.notch_filter(60); raw.filter(0.5, 40); raw.resample(200)
raw.set_eeg_reference("average")
mne.set_bipolar_reference(raw, anode=[...], cathode=[...])
raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False)

raw.plot(start=, duration=, scalings=dict(eeg=100e-6)); raw.compute_psd().plot()

ep = mne.make_fixed_length_epochs(raw, duration=4, overlap=2, preload=True)
ep = mne.Epochs(raw, events, event_id, tmin=-5, tmax=10, baseline=None, preload=True)
ep.metadata = df;  ep['label == "ictal"'];  ep.drop_bad(reject=dict(eeg=500e-6))
ep.get_data()                      # (n_epochs, canales, muestras)
ep.compute_psd(); ep.compute_tfr("morlet", freqs=, n_cycles=)

raw.save("x_raw.fif"); ep.save("x-epo.fif"); mne.read_epochs("x-epo.fif")
```

Tres reglas de oro:
1. MNE trabaja en **voltios**. Para tener µV, multiplica por `1e6` o usa `units="uV"`.
2. Casi todo modifica el objeto **in place**, así que usa `.copy()` cuando quieras conservar el original.
3. Filtra la señal **continua** antes de cortar ventanas, y valida siempre **separando por paciente**.
