"""T24: reference models for ds5 on the same temporal folds, with and without the two proxies.

Models (no tuning beyond the stated grid, selection on the fold validation macro F1 only, test read once):
  logistica : multinomial logistic regression on the network inputs (standardised one-hot + GB_* refitted
              per fold), sqrt class weights as in variant B; C chosen from {0.01, 0.1, 1.0} on validation.
  hgb       : HistGradientBoosting on the 101 one-hot columns (no GB_*), v3 hyperparameters and 114
              iterations, class_weight balanced; no tuning.
Each model with and without the pair dia_desv + dia_ult_menos_med (+ missing flag). The network figures come
from t20 (variants base and par, 5 seeds), so the same folds, data and proxies are compared.
Writes resultados_keras/t24_referencias.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402
import t18_ds5_gb_por_pliegue as t18  # noqa: E402
from t17_ds5_pliegues_temporales import PLIEGUES  # noqa: E402

PAR = ["dia_desv", "dia_ult_menos_med"]
GRID_C = [0.01, 0.1, 1.0]


def metr(y, pred, idx):
    f = f1_score(y, pred, average=None, labels=range(4), zero_division=0)
    return {"macro_f1": round(float(f.mean()), 4), "f1_leve": round(float(f[idx]), 4),
            "f1_clase": [round(float(v), 4) for v in f]}


def main():
    r3 = json.loads((t18.V3 / "resumen_v3.json").read_text(encoding="utf-8"))
    params, n_iter = r3["hiperparametros_elegidos"], int(r3["iteraciones"])
    X, y = t4.cargar_datos(t4.DS5)
    pesos, _ = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    K, idx = len(clases), clases.index("MORA_LEVE")
    d, F = t14.construir(estricto=True)
    orden = ["train", "val", "test"]
    pos = np.concatenate([np.where(d.Particion.to_numpy() == p)[0] for p in orden])
    d, F = d.iloc[pos].reset_index(drop=True), F.iloc[pos].reset_index(drop=True)
    yA = np.concatenate([y[p] for p in orden])
    Xo = np.vstack([pd.read_csv(t18.V2 / f"X_{p}_onehot.csv", float_precision="round_trip").fillna(0).to_numpy(float)
                    for p in orden])
    corte = d.Corte.to_numpy()
    cols = list(F.columns)
    n_oh = Xo.shape[1]

    res = {}
    for nombre, cf in PLIEGUES.items():
        m = {k: np.isin(corte, v) for k, v in cf.items()}
        gb = t18.gb_pliegue(Xo, yA, corte, m, params, n_iter, K)
        sc = StandardScaler().fit(np.hstack([Xo[m["train"]], gb["train"]]))
        XR = np.zeros((len(Xo), n_oh + len(t18.GB_COLS)), dtype=np.float64)
        for p in orden:
            XR[m[p]] = sc.transform(np.hstack([Xo[m[p]], gb[p]]))
        Fi = F.fillna(F[m["train"]].median())
        mu, sd = Fi[m["train"]].mean(), Fi[m["train"]].std().replace(0, 1.0)
        Zs = ((Fi - mu) / sd).to_numpy(float)
        falta = F.isna().to_numpy().astype(float)[:, [cols.index("n_fechas")]]
        Zpar = np.hstack([Zs[:, [cols.index(c) for c in PAR]], falta])
        Fpar = np.column_stack([Fi[PAR].to_numpy(float), falta])
        w = t4.pesos_sqrt_balanceados(yA[m["train"]], K).astype(float)[yA[m["train"]]]  # per-sample weights

        for var, usar in (("base", False), ("par", True)):
            # logistic regression on the network inputs
            M = np.hstack([XR, Zpar]) if usar else XR
            mejor = None
            for C in GRID_C:
                lr = LogisticRegression(C=C, max_iter=2000).fit(M[m["train"]], yA[m["train"]], sample_weight=w)
                v = f1_score(yA[m["val"]], lr.predict(M[m["val"]]), average="macro", labels=range(4), zero_division=0)
                if mejor is None or v > mejor[0]:
                    mejor = (v, C, lr)
            lr = mejor[2]
            res[f"{nombre}|logistica|{var}"] = {"C": mejor[1], "val_macro": round(float(mejor[0]), 4),
                                               "prueba": metr(yA[m["test"]], lr.predict(M[m["test"]]), idx)}
            # HGB on the one-hot columns
            Mh = np.hstack([Xo, Fpar]) if usar else Xo
            h = HistGradientBoostingClassifier(class_weight="balanced", random_state=42, max_iter=n_iter,
                                               early_stopping=False, **params).fit(Mh[m["train"]], yA[m["train"]])
            res[f"{nombre}|hgb|{var}"] = {
                "val_macro": round(float(f1_score(yA[m["val"]], h.predict(Mh[m["val"]]), average="macro",
                                                 labels=range(4), zero_division=0)), 4),
                "prueba": metr(yA[m["test"]], h.predict(Mh[m["test"]]), idx)}
            print(nombre, var, flush=True)

    # network figures from t20 (5-seed means per fold)
    t20 = json.loads((t4.SALIDA / "t20_subconjuntos_proxies.json").read_text(encoding="utf-8"))
    df = pd.DataFrame([{"pliegue": c["pliegue"], "var": c["variante"], "macro": c["macro_f1"], "leve": c["f1_clase"][idx]}
                       for c in t20["corridas"]])
    for nombre in PLIEGUES:
        for var in ("base", "par"):
            s = df[(df.pliegue == nombre) & (df["var"] == var)]
            res[f"{nombre}|red|{var}"] = {"prueba": {"macro_f1": round(float(s.macro.mean()), 4),
                                                    "f1_leve": round(float(s.leve.mean()), 4)}}

    resumen = {}
    for modelo in ("red", "logistica", "hgb"):
        for var in ("base", "par"):
            ks = [f"{n}|{modelo}|{var}" for n in PLIEGUES]
            resumen[f"{modelo}|{var}"] = {
                "macro_f1_medio": round(float(np.mean([res[k]["prueba"]["macro_f1"] for k in ks])), 4),
                "f1_leve_medio": round(float(np.mean([res[k]["prueba"]["f1_leve"] for k in ks])), 4),
                "macro_por_pliegue": [res[k]["prueba"]["macro_f1"] for k in ks]}
            print(modelo, var, resumen[f"{modelo}|{var}"], flush=True)
    (t4.SALIDA / "t24_referencias.json").write_text(json.dumps({"resumen": resumen, "detalle": res}, indent=2),
                                                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
