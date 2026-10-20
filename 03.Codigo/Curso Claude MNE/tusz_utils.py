"""
tusz_utils.py — Funciones de apoyo del curso de MNE con TUSZ.

Todo lo que hay aquí se construye paso a paso en los notebooks 01-07.
A partir del módulo 05 los notebooks importan estas funciones para no repetir código.

Rutas: se calculan a partir de la carpeta del proyecto (D:\\Dani\\TFM IA), así que
si mueves el proyecto entero sigue funcionando. Puedes forzarlas con las variables
de entorno TUSZ_ROOT, TUSZ_INVENTORY y TFM_SALIDAS.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
# Este fichero está en <PROYECTO>\03_codigo\<curso>\tusz_utils.py → subimos 2 niveles
PROJECT = Path(__file__).resolve().parents[2]          # D:\Dani\TFM IA

TUSZ_ROOT = Path(os.environ.get("TUSZ_ROOT", PROJECT / "00.TUSZ" / "v2.0.6"))


def _inventario_por_defecto() -> Path:
    """Busca el inventario en su sitio definitivo y, si no está, en 02.Notas."""
    for p in (PROJECT / "04_datos_derivados" / "inventario",
              PROJECT / "02.Notas" / "tusz_inventory_out"):
        if p.exists():
            return p
    return PROJECT / "04_datos_derivados" / "inventario"


INVENTORY = Path(os.environ.get("TUSZ_INVENTORY", _inventario_por_defecto()))

# Todo lo que generan los notebooks (.fif, cachés, ventanas .npy) va aquí, fuera del código
SALIDAS = Path(os.environ.get("TFM_SALIDAS", PROJECT / "04_datos_derivados" / "curso_MNE"))
SALIDAS.mkdir(parents=True, exist_ok=True)

# Fichero de ejemplo del curso: paciente con crisis de ausencia (absz), 3 crisis.
EJEMPLO_ABSZ = "aaaaabqv_s001_t000"

# ----------------------------------------------------------------------------
# Montaje TCP (Temporal Central Parasagittal) de 22 canales, en el orden de TUSZ
# ----------------------------------------------------------------------------
TCP_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),     # cadena temporal izq.
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),     # cadena temporal dcha.
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),      # cadena central
    ("C4", "T4"), ("T4", "A2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),     # parasagital izq.
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),     # parasagital dcha.
]
TCP_NAMES = [f"{a}-{c}" for a, c in TCP_PAIRS]

EEG_1020 = {"FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
            "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "OZ", "T1", "T2"}

SEIZURE_TYPES = {
    "fnsz": "Focal no específica", "gnsz": "Generalizada no específica",
    "spsz": "Focal simple parcial", "cpsz": "Focal compleja parcial",
    "absz": "Ausencia", "tnsz": "Tónica", "cnsz": "Clónica",
    "tcsz": "Tónico-clónica", "atsz": "Atónica", "mysz": "Mioclónica",
}


# ----------------------------------------------------------------------------
# Localizar y leer ficheros
# ----------------------------------------------------------------------------
def find_edf(file_id: str, root: Path = TUSZ_ROOT) -> Path:
    """'aaaaabqv_s001_t000' -> ruta completa al .edf (busca en train/dev/eval)."""
    patient = file_id.split("_")[0]
    hits = list((Path(root) / "edf").glob(f"*/{patient}/*/*/{file_id}.edf"))
    if not hits:
        raise FileNotFoundError(f"No encuentro {file_id}.edf bajo {root}")
    return hits[0]


def clean_name(ch: str) -> str:
    """'EEG FP1-LE' -> 'FP1', 'EEG T3-REF' -> 'T3', 'PHOTIC PH' -> 'PHOTIC'."""
    ch = re.sub(r"^EEG\s+", "", ch.strip())
    ch = re.sub(r"(-LE|-REF|\s+PH)$", "", ch)
    return ch.upper()


def guess_type(ch: str) -> str:
    if ch in EEG_1020:
        return "eeg"
    if ch.startswith(("EKG", "ECG")):
        return "ecg"
    if ch in {"ROC", "LOC"}:
        return "eog"
    if ch.startswith("EMG"):
        return "emg"
    return "misc"


def read_tusz_raw(file_or_path, preload: bool = True, annotate: bool = True,
                  verbose="error") -> mne.io.BaseRaw:
    """Lee un EDF de TUSZ, limpia nombres y tipos de canal y añade las anotaciones.

    - Nombres: 'EEG FP1-LE' -> 'FP1'
    - Tipos: eeg / ecg / eog / emg / misc
    - Anotaciones: crisis por tipo (del .csv, fusionando canales). Ver tusz_annotations().
    """
    path = Path(file_or_path)
    if path.suffix.lower() != ".edf":
        path = find_edf(str(file_or_path))
    raw = mne.io.read_raw_edf(path, preload=preload, verbose=verbose)
    mapping, seen = {}, set()
    for ch in raw.ch_names:
        new = clean_name(ch)
        if new in seen:          # evita duplicados raros
            new = ch
        seen.add(new)
        mapping[ch] = new
    raw.rename_channels(mapping)
    types = {ch: guess_type(ch) for ch in raw.ch_names}
    try:
        raw.set_channel_types(types, on_unit_change="ignore")
    except TypeError:  # versiones antiguas de MNE
        raw.set_channel_types(types)
    if annotate:
        raw.set_annotations(tusz_annotations(path, kind="multi"))
    return raw


def get_montage():
    """Montaje 10-20 estándar (nombre nuevo en MNE>=1.13, viejo como respaldo)."""
    for name in ("colin27_1020", "standard_1020"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return mne.channels.make_standard_montage(name)
        except ValueError:
            continue
    raise RuntimeError("No hay montaje 10-20 disponible en esta versión de MNE")


def set_1020(raw, drop_missing=True):
    """Asigna posiciones 10-20 a los canales EEG (sin distinguir mayúsculas).

    drop_missing: elimina los canales EEG que no tienen posición estándar (p. ej. T1, T2),
    necesario para los mapas topográficos.
    """
    raw.set_montage(get_montage(), match_case=False, on_missing="ignore")
    if drop_missing:
        sin_pos = [ch["ch_name"] for ch in raw.info["chs"]
                   if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH and np.isnan(ch["loc"][:3]).any()
                   or (ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH and not np.any(ch["loc"][:3]))]
        if sin_pos:
            raw.drop_channels(sin_pos)
    return raw


# ----------------------------------------------------------------------------
# Anotaciones
# ----------------------------------------------------------------------------
def read_tusz_csv(path) -> pd.DataFrame:
    """Lee un .csv o .csv_bi de TUSZ (las líneas que empiezan por # son cabecera)."""
    return pd.read_csv(path, comment="#")


def merge_intervals(df: pd.DataFrame, gap: float = 0.0) -> pd.DataFrame:
    """Fusiona intervalos solapados (por etiqueta). Útil porque en el .csv cada canal
    tiene su propio inicio/fin y queremos UNA anotación por crisis."""
    out = []
    for label, g in df.sort_values("start_time").groupby("label"):
        cur_s, cur_e = None, None
        for s, e in zip(g.start_time, g.stop_time):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e + gap:
                cur_e = max(cur_e, e)
            else:
                out.append((cur_s, cur_e, label))
                cur_s, cur_e = s, e
        if cur_s is not None:
            out.append((cur_s, cur_e, label))
    return pd.DataFrame(out, columns=["start_time", "stop_time", "label"]).sort_values("start_time")


def tusz_annotations(edf_path, kind: str = "multi", keep_bckg: bool = False,
                     per_channel: bool = False) -> mne.Annotations:
    """Convierte las anotaciones de TUSZ en mne.Annotations.

    kind='bi'    -> usa .csv_bi : 'seiz' / 'bckg' (crisis sí/no, todos los canales)
    kind='multi' -> usa .csv    : tipo de crisis ('absz', 'fnsz', ...)
    per_channel  -> (solo multi) una anotación por canal TCP, con ch_names=...
                    Solo vale si el Raw tiene canales TCP ('FP1-F7', ...).
    """
    edf_path = Path(edf_path)
    csv = edf_path.with_suffix(".csv_bi" if kind == "bi" else ".csv")
    df = read_tusz_csv(csv)
    if not keep_bckg:
        df = df[df.label != "bckg"]
    if df.empty:
        return mne.Annotations([], [], [])
    if per_channel and kind == "multi":
        return mne.Annotations(onset=df.start_time.values,
                               duration=(df.stop_time - df.start_time).values,
                               description=df.label.values,
                               ch_names=[[c] for c in df.channel])
    if kind == "multi":
        df = merge_intervals(df)
    return mne.Annotations(onset=df.start_time.values,
                           duration=(df.stop_time - df.start_time).values,
                           description=df.label.values)


def seizure_intervals(raw, prefix_exclude=("BAD", "bckg")) -> list[tuple[float, float, str]]:
    """Lista de (inicio, fin, etiqueta) de las crisis anotadas en raw."""
    return [(a["onset"] - raw.first_time, a["onset"] - raw.first_time + a["duration"], a["description"])
            for a in raw.annotations if not a["description"].startswith(prefix_exclude)]


# ----------------------------------------------------------------------------
# Montaje TCP y preprocesado
# ----------------------------------------------------------------------------
def make_tcp(raw, keep_annotations: bool = True) -> mne.io.BaseRaw:
    """Raw referencial (LE o AR) -> Raw bipolar TCP (22 canales, o 20 si faltan A1/A2)."""
    pairs = [(a, c) for a, c in TCP_PAIRS if a in raw.ch_names and c in raw.ch_names]
    anodes, cathodes = zip(*pairs)
    tcp = mne.set_bipolar_reference(raw.copy().pick("eeg"), anode=list(anodes),
                                    cathode=list(cathodes), drop_refs=True, verbose="error")
    names = [f"{a}-{c}" for a, c in pairs]
    tcp.pick(names)                 # descarta T1, T2, FZ, ... y deja el orden TUSZ
    if not keep_annotations:
        tcp.set_annotations(None)
    return tcp


def preprocess(raw, l_freq=0.5, h_freq=40.0, notch=60.0, sfreq=200.0, tcp=True):
    """Pipeline estándar del curso: TCP -> notch 60 Hz -> paso banda -> remuestreo."""
    out = make_tcp(raw) if tcp else raw.copy().pick("eeg")
    if notch and notch < out.info["sfreq"] / 2:
        out.notch_filter(notch, verbose="error")
    out.filter(l_freq, h_freq, verbose="error")
    if sfreq and out.info["sfreq"] != sfreq:
        out.resample(sfreq, verbose="error")
    return out


# ----------------------------------------------------------------------------
# Ventanas (epochs) etiquetadas
# ----------------------------------------------------------------------------
def label_windows(starts, win, seizures, preictal=None, sph=0.0, postictal=0.0):
    """Etiqueta cada ventana [s, s+win) a partir de los intervalos de crisis.

    Devuelve un DataFrame con:
      ictal_frac  fracción de la ventana dentro de crisis
      label       'ictal' | 'preictal' | 'postictal' | 'interictal'
      sz_type     tipo de crisis (si ictal)
      t_to_next   segundos hasta la siguiente crisis (NaN si no hay)
    preictal: duración (s) del periodo preictal antes del inicio de cada crisis
    sph: horizonte de predicción (seizure prediction horizon) entre preictal y crisis
    """
    rows = []
    for s in starts:
        e = s + win
        frac, typ = 0.0, None
        for a, b, lab in seizures:
            ov = max(0.0, min(e, b) - max(s, a))
            if ov > frac * win:
                frac, typ = ov / win, lab
        nxt = [a - e for a, b, _ in seizures if a >= e]
        t_next = min(nxt) if nxt else np.nan
        prev = [s - b for a, b, _ in seizures if b <= s]
        t_prev = min(prev) if prev else np.nan
        if frac > 0:
            lab = "ictal"
        elif preictal and not np.isnan(t_next) and sph <= t_next < sph + preictal:
            lab = "preictal"
        elif postictal and not np.isnan(t_prev) and t_prev < postictal:
            lab = "postictal"
        else:
            lab = "interictal"
        rows.append(dict(start=s, ictal_frac=frac, label=lab, sz_type=typ,
                         t_to_next=t_next, t_since_prev=t_prev))
    return pd.DataFrame(rows)


def make_windows(raw, win=4.0, step=2.0, **label_kw) -> mne.Epochs:
    """Corta Raw en ventanas de `win` s con paso `step` s y añade metadata con etiquetas."""
    sz = seizure_intervals(raw)
    epochs = mne.make_fixed_length_epochs(raw, duration=win, overlap=win - step,
                                          preload=True, reject_by_annotation=False,
                                          verbose="error")
    starts = epochs.events[:, 0] / raw.info["sfreq"] - raw.first_time
    epochs.metadata = label_windows(starts, win, sz, **label_kw)
    return epochs


BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def band_power(epochs, bands=BANDS, relative=True, log=True):
    """Potencia por banda -> array (n_epochs, n_canales, n_bandas)."""
    fmin = min(b[0] for b in bands.values())
    fmax = max(b[1] for b in bands.values())
    spec = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose="error")
    psd, freqs = spec.get_data(return_freqs=True)
    bp = np.stack([psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) for lo, hi in bands.values()], -1)
    if relative:
        bp = bp / bp.sum(-1, keepdims=True)
    if log:
        bp = np.log10(bp + 1e-20)
    return bp
