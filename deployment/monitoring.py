"""Utilitas monitoring: data quality, drift (PSI), performa model, dan keputusan alert."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4
PSI_WARN, PSI_ALERT = 0.10, 0.25          # konvensi umum industri
MISSING_DELTA_WARN = 0.10                 # kenaikan missing rate > 10 poin persentase
UNSEEN_CAT_WARN = 0.05                    # >5% nilai kategori belum pernah terlihat


def _psi(ref_p, cur_p):
    ref_p = np.clip(np.asarray(ref_p, float), EPS, None)
    cur_p = np.clip(np.asarray(cur_p, float), EPS, None)
    ref_p, cur_p = ref_p / ref_p.sum(), cur_p / cur_p.sum()
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _status(psi):
    return "ALERT" if psi >= PSI_ALERT else "WARN" if psi >= PSI_WARN else "OK"


def build_reference(X, numeric, categorical, min_share=0.02):
    """Ringkasan distribusi data training -> dict (disimpan sebagai JSON)."""
    ref = {"n_rows": int(len(X)), "numeric": {}, "categorical": {}}
    for c in numeric:
        s = X[c].dropna().astype(float)
        edges = np.unique(np.quantile(s, np.linspace(0, 1, 11)))
        inner = edges[1:-1]
        if len(inner) < 2:        # fitur biner / hampir konstan: pakai nilai unik sebagai bin
            vals = np.sort(s.value_counts().head(10).index.values.astype(float))
            inner = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else np.array([])
        counts = np.bincount(np.digitize(s, inner), minlength=len(inner) + 1)
        ref["numeric"][c] = {"inner_edges": inner.tolist(),
                             "bin_props": (counts / counts.sum()).tolist(),
                             "missing_rate": float(X[c].isna().mean()),
                             "mean": float(s.mean()), "p01": float(s.quantile(.01)),
                             "p99": float(s.quantile(.99))}
    for c in categorical:
        vc = X[c].fillna("Unknown").value_counts()
        known = vc[vc / len(X) >= min_share]   # <=~50 bin; bin terlalu banyak membuat PSI berisik
        props = (known / len(X)).to_dict()
        props["__other__"] = float(1 - sum(props.values()))
        ref["categorical"][c] = {"props": props, "all_values": vc.index.tolist(),
                                 "missing_rate": float(X[c].isna().mean())}
    return ref


def drift_report(batch, ref):
    rows = []
    for c, r in ref["numeric"].items():
        s = pd.to_numeric(batch[c], errors="coerce")
        v = s.dropna()
        if len(v) == 0 or len(r["inner_edges"]) == 0:
            psi = np.nan
        else:
            counts = np.bincount(np.digitize(v, r["inner_edges"]), minlength=len(r["bin_props"]))
            psi = _psi(r["bin_props"], counts / counts.sum())
        rows.append({"feature": c, "type": "numeric", "psi": psi,
                     "missing_rate": float(s.isna().mean()),
                     "missing_delta": float(s.isna().mean() - r["missing_rate"]),
                     "unseen_rate": np.nan,
                     "out_of_range_rate": float(((v < r["p01"]) | (v > r["p99"])).mean()) if len(v) else np.nan})
    for c, r in ref["categorical"].items():
        s = batch[c].fillna("Unknown")
        known_all = set(r["all_values"])
        mapped = s.where(s.isin(r["props"].keys()), "__other__")
        keys = list(r["props"].keys())
        cur = mapped.value_counts(normalize=True).reindex(keys).fillna(0).values
        rows.append({"feature": c, "type": "categorical",
                     "psi": _psi([r["props"][k] for k in keys], cur),
                     "missing_rate": float(batch[c].isna().mean()),
                     "missing_delta": float(batch[c].isna().mean() - r["missing_rate"]),
                     "unseen_rate": float((~s.isin(known_all)).mean()),
                     "out_of_range_rate": np.nan})
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].map(lambda p: "n/a" if pd.isna(p) else _status(p))
    flag = (out["missing_delta"] > MISSING_DELTA_WARN) | (out["unseen_rate"] > UNSEEN_CAT_WARN)
    out.loc[flag & (out["status"] == "OK"), "status"] = "WARN"
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def performance_report(y_true_log, y_pred_log):
    y_true_log, y_pred_log = np.asarray(y_true_log), np.asarray(y_pred_log)
    res = y_true_log - y_pred_log
    return {"n": int(len(res)),
            "rmse_log": float(np.sqrt(np.mean(res ** 2))),
            "bias_log": float(res.mean()),                     # >0 = model under-predict
            "resid_sd": float(res.std(ddof=1)) if len(res) > 1 else 0.0,
            "mae_usd": float(np.mean(np.abs(np.expm1(y_true_log) - np.expm1(y_pred_log))))}


LEVELS = ["OK", "WARN", "ALERT", "RETRAIN"]


def overall_status(drift_df, perf, baseline_rmse_log, rmse_warn=1.10, rmse_alert=1.25,
                   bias_warn=0.10, bias_alert=0.25, min_n=30):
    """Satu keputusan ringkas: OK < WARN < ALERT (investigasi) < RETRAIN.

    - Drift fitur saja -> maksimal ALERT: drift belum tentu menurunkan akurasi, perlu dikonfirmasi harga aktual.
    - Performa turun (RMSE naik, atau bias sistematis yang signifikan) -> RETRAIN.
    - Performa mulai turun + drift berat -> RETRAIN.
    Bias dicek terpisah karena RMSE kurang sensitif terhadap pergeseran harga yang seragam.
    """
    reasons, lvl = [], 0
    n_alert = int((drift_df["status"] == "ALERT").sum())
    n_warn = int((drift_df["status"] == "WARN").sum())
    if n_alert:
        lvl = 2; reasons.append(f"{n_alert} fitur drift berat (PSI>={PSI_ALERT})")
    if n_warn:
        lvl = max(lvl, 1); reasons.append(f"{n_warn} fitur drift ringan/data quality")

    perf_lvl = 0
    if perf is not None and perf["n"] >= min_n:
        ratio = perf["rmse_log"] / baseline_rmse_log
        se = perf.get("resid_sd", 0) / np.sqrt(perf["n"]) if perf.get("resid_sd") else np.inf
        bias_sig = abs(perf["bias_log"]) / se >= 3          # bias harus signifikan secara statistik
        if ratio >= rmse_alert:
            perf_lvl = 2; reasons.append(f"RMSE(log) {ratio:.2f}x baseline")
        elif ratio >= rmse_warn:
            perf_lvl = 1; reasons.append(f"RMSE(log) {ratio:.2f}x baseline")
        if bias_sig and abs(perf["bias_log"]) >= bias_alert:
            perf_lvl = 2; reasons.append(f"bias {perf['bias_log']:+.2f} (log) signifikan")
        elif bias_sig and abs(perf["bias_log"]) >= bias_warn:
            perf_lvl = max(perf_lvl, 1); reasons.append(f"bias {perf['bias_log']:+.2f} (log) signifikan")

    if perf_lvl == 2 or (perf_lvl == 1 and n_alert):
        lvl = 3
    elif perf_lvl == 1:
        lvl = max(lvl, 1)
    return {"status": LEVELS[lvl], "reasons": reasons}


def load_logs(log_dir):
    """Gabungkan log prediksi + feedback aktual (join lewat request_id)."""
    log_dir = Path(log_dir)
    def _read(p):
        if not p.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
    preds, acts = _read(log_dir / "predictions.jsonl"), _read(log_dir / "actuals.jsonl")
    if len(preds) and len(acts):
        preds = preds.merge(acts[["request_id", "actual_price_usd"]], on="request_id", how="left")
    return preds
