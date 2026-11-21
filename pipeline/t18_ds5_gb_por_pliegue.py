"""T18: ds5 temporal folds with GB_* refitted per fold, plus confusion matrices.

Recipe copied from datasets_multiclase_v3/generar_v3.py (hyperparameters from resumen_v3.json, no retuning):
HistGradientBoosting on the 101 one-hot columns; train GB_* from GroupKFold(5) out-of-fold probabilities
grouped by Corte; validation and test GB_* from the model fitted on the whole fold train; the doubt-zone
threshold tau comes from the fold validation. The network input is StandardScaler([one-hot, GB_*]) fitted on
the fold train, then the payment-timing proxies (standardised the same way).

Two modes per fold, 5 seeds, variant B (sqrt-weighted CE):
  gb_original: GB_* as stored in the v4 files (what t17 used).
  gb_refit:    GB_* refitted on the fold train (no information from later cuts or from other partitions).
Checks: (1) on fold F3 the refit GB_* must match the stored ones; (2) the standardised one-hot part must match
X_train_nn. Confusion matrices (counts summed over seeds) are stored per fold, mode and variant.
Writes resultados_keras/t18_gb_por_pliegue.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402
from t17_ds5_pliegues_temporales import PLIEGUES, SEMILLAS  # noqa: E402

V2 = t14.BI / "datasets_multiclase_v2/salida/ds5_pagos_multimes"
V3 = t14.BI / "datasets_multiclase_v3/ds5_pagos_multimes"
COBERTURA = 0.70
GB_COLS = ["GB_P_AL_DIA", "GB_P_MORA_LEVE", "GB_P_MORA_31_90", "GB_P_MORA_MAYOR_90", "GB_Margen", "GB_Entropia",
           "GB_Zona_Duda", "GB_Esperanza", "GB_P_Mayor_0", "GB_P_Mayor_1", "GB_P_Mayor_2"]


def hgb(params, n_iter):
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=42, max_iter=n_iter,
                                          early_stopping=False, **params)


def features_gb(P, tau, K):
    pred, srt = P.argmax(1), np.sort(P, axis=1)
    return np.column_stack([
        P, srt[:, -1] - srt[:, -2], -(P * np.log(np.clip(P, 1e-12, None))).sum(1) / np.log(K),
        (P.max(1) < tau[pred]).astype(float), P @ np.arange(K),
        *[P[:, k + 1:].sum(1) for k in range(K - 1)]])


def gb_pliegue(Xo, y, corte, m, params, n_iter, K):
    tr = np.where(m["train"])[0]
    P = {"train": np.zeros((len(tr), K))}
    # F1 has only 4 train cuts, so GroupKFold uses min(5, cuts) folds (deviation from the v3 recipe)
    for a, b in GroupKFold(min(5, len(np.unique(corte[tr])))).split(tr, groups=corte[tr]):
        P["train"][b] = hgb(params, n_iter).fit(Xo[tr[a]], y[tr[a]]).predict_proba(Xo[tr[b]])
    full = hgb(params, n_iter).fit(Xo[tr], y[tr])
    P["val"], P["test"] = full.predict_proba(Xo[m["val"]]), full.predict_proba(Xo[m["test"]])
    cv, mv = P["val"].argmax(1), P["val"].max(1)
    tau = np.array([np.quantile(mv[cv == k], 1 - COBERTURA) if (cv == k).sum() >= 20 else 0.0 for k in range(K)])
    return {p: features_gb(P[p], tau, K) for p in P}


def main():
    r3 = json.loads((V3 / "resumen_v3.json").read_text(encoding="utf-8"))
    params, n_iter = r3["hiperparametros_elegidos"], int(r3["iteraciones"])
    X, y = t4.cargar_datos(t4.DS5)
    pesos, cfg = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    K, idx = len(clases), clases.index("MORA_LEVE")
    d, F = t14.construir(estricto=True)
    orden = ["train", "val", "test"]
    pos = np.concatenate([np.where(d.Particion.to_numpy() == p)[0] for p in orden])
    d, F = d.iloc[pos].reset_index(drop=True), F.iloc[pos].reset_index(drop=True)
    XA = np.vstack([X[p] for p in orden])
    yA = np.concatenate([y[p] for p in orden])
    Xo = np.vstack([pd.read_csv(V2 / f"X_{p}_onehot.csv", float_precision="round_trip").fillna(0).to_numpy(float)
                    for p in orden])
    corte = d.Corte.to_numpy()
    cols = list(F.columns)
    n_oh = Xo.shape[1]
    print("one-hot", n_oh, "red", XA.shape[1], flush=True)

    # check 2: standardised one-hot part equals the stored network input (original train)
    mtr0 = d.Particion.to_numpy() == "train"
    sc0 = StandardScaler().fit(Xo[mtr0])
    paridad_oh = float(np.abs(sc0.transform(Xo) - XA[:, :n_oh]).max())
    print("paridad one-hot vs X_nn (max abs):", round(paridad_oh, 6), flush=True)

    res = {"paridad_onehot_max_abs": paridad_oh, "clases": clases, "corridas": [], "confusion": {}}
    w_all = None
    for nombre, cf in PLIEGUES.items():
        m = {k: np.isin(corte, v) for k, v in cf.items()}
        gb = gb_pliegue(Xo, yA, corte, m, params, n_iter, K)
        if nombre == "F3":  # check 1: refit GB vs stored GB on the original partition
            stored = {p: XA[m[p]][:, n_oh:] for p in orden}
            # stored GB inputs are standardised with the original train statistics; apply the same step
            sc_st = StandardScaler().fit(gb["train"])
            comp = {p: float(np.abs(sc_st.transform(gb[p]) - stored[p]).max()) for p in orden}
            res["paridad_gb_F3_max_abs_estandarizado"] = comp
            print("paridad GB refit vs guardado (F3):", {k: round(v, 4) for k, v in comp.items()}, flush=True)
        Z_med = F[m["train"]].median()
        Fi = F.fillna(Z_med)
        mu, sd = Fi[m["train"]].mean(), Fi[m["train"]].std().replace(0, 1.0)
        falta = F.isna().to_numpy().astype(np.float32)[:, [cols.index("n_fechas")]]
        Zp = np.hstack([((Fi - mu) / sd).to_numpy(np.float32), falta])

        sc = StandardScaler().fit(np.hstack([Xo[m["train"]], gb["train"]]))
        XR = np.zeros((len(Xo), n_oh + len(GB_COLS)), dtype=np.float32)
        for p in orden:
            XR[m[p]] = sc.transform(np.hstack([Xo[m[p]], gb[p]]))
        modos = {"gb_original": XA, "gb_refit": XR}
        w = t4.pesos_sqrt_balanceados(yA[m["train"]], K)
        print(nombre, {k: int(v.sum()) for k, v in m.items()}, flush=True)
        for modo, M0 in modos.items():
            variantes = {"base": M0, "proxies": np.hstack([M0, Zp])}
            conf = {v: np.zeros((K, K), int) for v in variantes}
            filas = []
            for semilla in SEMILLAS:
                r = {"pliegue": nombre, "modo": modo, "semilla": semilla}
                for var, M in variantes.items():
                    mod = t4.entrenar(M[m["train"]], yA[m["train"]], M[m["val"]], yA[m["val"]], K, cfg["alpha"],
                                      "wce", semilla, pesos_clase=w)
                    P = t4.probas(mod, M[m["test"]])
                    r[var] = t4.metricas(yA[m["test"]], P)
                    conf[var] += confusion_matrix(yA[m["test"]], P.argmax(1), labels=range(K))
                r["d_macro"] = round(r["proxies"]["macro_f1"] - r["base"]["macro_f1"], 4)
                r["d_leve"] = round(r["proxies"]["f1_clase"][idx] - r["base"]["f1_clase"][idx], 4)
                filas.append(r)
                print(nombre, modo, semilla, "d_macro", r["d_macro"], "d_leve", r["d_leve"], flush=True)
            res["corridas"] += filas
            res["confusion"][f"{nombre}|{modo}"] = {v: c.tolist() for v, c in conf.items()}

    def resumen(sel):
        dm, dl = np.array([f["d_macro"] for f in sel]), np.array([f["d_leve"] for f in sel])
        return {"n": len(sel), "d_macro_media": round(float(dm.mean()), 4), "d_macro_desv": round(float(dm.std()), 4),
                "d_macro_positivos": int((dm > 0).sum()), "d_leve_media": round(float(dl.mean()), 4),
                "d_leve_desv": round(float(dl.std()), 4), "d_leve_positivos": int((dl > 0).sum()),
                "base_macro": round(float(np.mean([f["base"]["macro_f1"] for f in sel])), 4),
                "proxies_macro": round(float(np.mean([f["proxies"]["macro_f1"] for f in sel])), 4),
                "base_leve": round(float(np.mean([f["base"]["f1_clase"][idx] for f in sel])), 4),
                "proxies_leve": round(float(np.mean([f["proxies"]["f1_clase"][idx] for f in sel])), 4)}

    res["resumen"] = {f"{n}|{mo}": resumen([f for f in res["corridas"] if f["pliegue"] == n and f["modo"] == mo])
                      for n in PLIEGUES for mo in ("gb_original", "gb_refit")}
    res["global"] = {mo: resumen([f for f in res["corridas"] if f["modo"] == mo]) for mo in ("gb_original", "gb_refit")}
    for k, v in res["resumen"].items():
        print(k, v)
    print("global", res["global"])
    (t4.SALIDA / "t18_gb_por_pliegue.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
