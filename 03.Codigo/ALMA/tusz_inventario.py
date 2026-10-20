"""
Inventario del dataset TUSZ v2.0.6  (TFM - Fase 1)

Recorre edf/{train,dev,eval} y genera en la carpeta de salida:
  - files.csv     : una fila por EDF (montaje, canales, fs, duración, nº crisis, tipos)
  - seizures.csv  : una fila por crisis del .csv_bi, con su tipo y su lead-in
  - summary.txt   : resumen por split, tipos de crisis y viabilidad para predicción

No usa MNE: lee solo la cabecera de cada EDF, así que tarda pocos minutos.

Uso:
    python tusz_inventario.py
    python tusz_inventario.py --root "D:\\Dani\\TFM IA\\00.TUSZ\\v2.0.6" --out "D:\\...\\inventario"
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]          # D:\Dani\TFM IA
ROOT_DEF = PROJECT / "00.TUSZ" / "v2.0.6"
OUT_DEF = PROJECT / "04_datos_derivados" / "inventario"
SPLITS = ["train", "dev", "eval"]


# ---------------------------------------------------------------- lectura
def edf_header(path):
    """Devuelve (n_señales, fs de la 1ª señal, duración en s) leyendo solo la cabecera EDF."""
    with open(path, "rb") as f:
        h = f.read(256)
        n_rec = int(h[236:244])
        rec_dur = float(h[244:252])
        ns = int(h[252:256])
        f.seek(256 + ns * 216)                 # salta hasta 'nº muestras por registro'
        spr = [int(f.read(8)) for _ in range(ns)]
    return ns, spr[0] / rec_dur, n_rec * rec_dur


def read_ann(path):
    """Lee un .csv / .csv_bi de TUSZ -> (duración declarada, DataFrame de anotaciones)."""
    dur = None
    with open(path) as f:
        for line in f:
            if line.startswith("# duration"):
                dur = float(line.split("=")[1].split()[0])
            if not line.startswith("#"):
                break
    df = pd.read_csv(path, comment="#")
    return dur, df


def dominant_label(multi, start, stop):
    """Tipo de crisis con más solapamiento (sumado sobre canales) con [start, stop]."""
    m = multi[multi.label != "bckg"]
    ov = (m.stop_time.clip(upper=stop) - m.start_time.clip(lower=start)).clip(lower=0)
    ov = ov.groupby(m.label).sum()
    ov = ov[ov > 0]
    return ov.idxmax() if len(ov) else "sin_tipo"


# ---------------------------------------------------------------- inventario
def build(root):
    files, seizures = [], []
    for split in SPLITS:
        edfs = sorted((root / "edf" / split).glob("*/*/*/*.edf"))
        print(f"{split}: {len(edfs)} EDF")
        for edf in edfs:
            patient, session, token = edf.stem.split("_")
            ns, fs, edf_dur = edf_header(edf)
            csv_dur, multi = read_ann(edf.with_suffix(".csv"))
            _, bi = read_ann(edf.with_suffix(".csv_bi"))
            bi = bi[bi.label == "seiz"]
            labels = sorted(set(multi.label) - {"bckg"})
            files.append(dict(
                split=split, patient=patient, session=session, token=token,
                montage=edf.parent.name, file=edf.stem, n_signals=ns, fs=fs,
                edf_dur_s=edf_dur, csv_dur_s=csv_dur,
                n_seiz_events_bi=len(bi),
                seiz_secs_bi=round((bi.stop_time - bi.start_time).sum(), 2),
                labels=";".join(labels) if labels else None))
            for _, r in bi.iterrows():
                seizures.append(dict(
                    split=split, patient=patient, session=session, token=token,
                    file=edf.stem, label=dominant_label(multi, r.start_time, r.stop_time),
                    start=r.start_time, stop=r.stop_time, file_dur=edf_dur))
    return pd.DataFrame(files), pd.DataFrame(seizures)


def add_leadin(files, seiz):
    """Tiempo absoluto dentro de la sesión (tokens t000, t001... encadenados)
    y lead-in = tiempo sin crisis antes de cada crisis dentro de la sesión."""
    f = files.sort_values(["split", "patient", "session", "token"]).copy()
    f["offset"] = f.groupby(["split", "patient", "session"]).edf_dur_s.cumsum() - f.edf_dur_s
    seiz = seiz.merge(f[["file", "offset"]], on="file")
    seiz["abs_start"] = seiz.start + seiz.offset
    seiz["abs_stop"] = seiz.stop + seiz.offset
    seiz = seiz.sort_values(["split", "patient", "session", "abs_start"])
    prev_stop = seiz.groupby(["split", "patient", "session"]).abs_stop.shift().fillna(0)
    seiz["leadin_s"] = (seiz.abs_start - prev_stop).round(1)
    return seiz.drop(columns="offset").reset_index(drop=True)


# ---------------------------------------------------------------- resumen
def summary(files, seiz, root):
    L = ["=== INVENTARIO TUSZ ===", f"Raiz: {root}", ""]
    rows, pats = [], {}
    for s in SPLITS:
        d = files[files.split == s]
        pats[s] = set(d.patient)
        rows.append({"split": s, "pacientes": d.patient.nunique(),
                     "sesiones": d[["patient", "session"]].drop_duplicates().shape[0],
                     "ficheros": len(d), "horas": round(d.edf_dur_s.sum() / 3600, 1),
                     "pac.c/crisis": d[d.n_seiz_events_bi > 0].patient.nunique(),
                     "eventos": d.n_seiz_events_bi.sum(),
                     "min.crisis": round(d.seiz_secs_bi.sum() / 60, 1)})
    L.append(pd.DataFrame(rows).to_string(index=False))
    L += ["", "Pacientes compartidos entre splits (debe ser 0): "
          f"train/dev={len(pats['train'] & pats['dev'])}, "
          f"train/eval={len(pats['train'] & pats['eval'])}, "
          f"dev/eval={len(pats['dev'] & pats['eval'])}", ""]
    L.append(f"Montajes:  {dict(Counter(files.montage))}")
    L.append(f"Frecuencias de muestreo (Hz):  {dict(Counter(files.fs.round().astype(int)))}")
    L.append(f"N. de senales por EDF:  {dict(Counter(files.n_signals))}")

    L += ["", "--- Tipos de crisis (1 evento = 1 evento del .csv_bi; "
          "tipo = etiqueta multiclase dominante) ---"]
    seiz = seiz.assign(dur=seiz.stop - seiz.start)
    rows = []
    for t in sorted(seiz.label.unique()):
        row = {"tipo": t}
        for s in SPLITS:
            d = seiz[(seiz.label == t) & (seiz.split == s)]
            row.update({f"{s}_ev": len(d), f"{s}_pac": d.patient.nunique(),
                        f"{s}_min": round(d.dur.sum() / 60, 1)})
        rows.append(row)
    L.append(pd.DataFrame(rows).to_string(index=False))
    L += ["", "--- Viabilidad para PREDICCION (lead-in = tiempo sin crisis previo dentro de la sesion) ---",
          "(aproximacion: se agrupan los tokens t000, t001... de la sesion como consecutivos)"]
    rows = []
    for s in SPLITS:
        d = seiz[seiz.split == s]
        a, b = d[d.leadin_s >= 1800], d[d.leadin_s >= 3600]
        rows.append({"split": s, "crisis": len(d), "lead>=30min": len(a), "pac_30": a.patient.nunique(),
                     "lead>=60min": len(b), "pac_60": b.patient.nunique()})
    L.append(pd.DataFrame(rows).to_string(index=False))
    du = files.edf_dur_s
    L += ["", f"Duracion de fichero: min/mediana/max (s): {du.min():.0f} / {du.median():.0f} / {du.max():.0f}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Inventario de TUSZ")
    ap.add_argument("--root", type=Path, default=ROOT_DEF, help="carpeta v2.0.6 de TUSZ")
    ap.add_argument("--out", type=Path, default=OUT_DEF, help="carpeta de salida")
    a = ap.parse_args()
    assert (a.root / "edf").exists(), f"No encuentro {a.root / 'edf'}"
    a.out.mkdir(parents=True, exist_ok=True)

    files, seiz = build(a.root)
    seiz = add_leadin(files, seiz)
    files.to_csv(a.out / "files.csv", index=False)
    seiz.to_csv(a.out / "seizures.csv", index=False)
    txt = summary(files, seiz, a.root)
    (a.out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("Guardado en", a.out)


if __name__ == "__main__":
    main()
