"""T19: ablation of the ds5 payment-timing proxies inside the Keras network (GB_* refitted per fold).

Variants (variant B, sqrt-weighted CE, 3 temporal folds x 5 seeds, same data as t18 gb_refit):
  base            : 112 inputs.
  full            : base + 9 proxies + missing flag.
  sin_<proxy>     : full minus one proxy (9 variants).
  solo_dia_desv   : base + dia_desv + missing flag.
The missing flag (n_fechas is NaN) is always kept when any proxy is present.
Reported per variant: test macro F1 and MORA_LEVE F1 means, difference against base and against full,
and in how many of the 15 fold x seed pairs the variant beats `full`. Writes
resultados_keras/t19_ablacion_proxies.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402
import t18_ds5_gb_por_pliegue as t18  # noqa: E402
from t17_ds5_pliegues_temporales import PLIEGUES, SEMILLAS  # noqa: E402


def main():
    r3 = json.loads((t18.V3 / "resumen_v3.json").read_text(encoding="utf-8"))
    params, n_iter = r3["hiperparametros_elegidos"], int(r3["iteraciones"])
    X, y = t4.cargar_datos(t4.DS5)
    pesos, cfg = t4.cargar_config(t4.DS5)
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

    variantes = {"base": [], "full": cols}
    for c in cols:
        variantes[f"sin_{c}"] = [k for k in cols if k != c]
    variantes["solo_dia_desv"] = ["dia_desv"]

    corridas = []
    for nombre, cf in PLIEGUES.items():
        m = {k: np.isin(corte, v) for k, v in cf.items()}
        gb = t18.gb_pliegue(Xo, yA, corte, m, params, n_iter, K)
        sc = StandardScaler().fit(np.hstack([Xo[m["train"]], gb["train"]]))
        XR = np.zeros((len(Xo), n_oh + len(t18.GB_COLS)), dtype=np.float32)
        for p in orden:
            XR[m[p]] = sc.transform(np.hstack([Xo[m[p]], gb[p]]))
        Fi = F.fillna(F[m["train"]].median())
        mu, sd = Fi[m["train"]].mean(), Fi[m["train"]].std().replace(0, 1.0)
        Zs = ((Fi - mu) / sd).to_numpy(np.float32)
        falta = F.isna().to_numpy().astype(np.float32)[:, [cols.index("n_fechas")]]
        w = t4.pesos_sqrt_balanceados(yA[m["train"]], K)
        print(nombre, flush=True)
        for var, usar in variantes.items():
            M = XR if not usar else np.hstack([XR, Zs[:, [cols.index(c) for c in usar]], falta])
            for semilla in SEMILLAS:
                mod = t4.entrenar(M[m["train"]], yA[m["train"]], M[m["val"]], yA[m["val"]], K, cfg["alpha"], "wce",
                                  semilla, pesos_clase=w)
                r = t4.metricas(yA[m["test"]], t4.probas(mod, M[m["test"]]))
                corridas.append({"pliegue": nombre, "variante": var, "semilla": semilla, "macro_f1": r["macro_f1"],
                                 "f1_clase": r["f1_clase"]})
            print(nombre, var, flush=True)

    df = pd.DataFrame([{"pliegue": c["pliegue"], "variante": c["variante"], "semilla": c["semilla"],
                        "macro": c["macro_f1"], "leve": c["f1_clase"][idx]} for c in corridas])
    piv_m = df.pivot_table(index=["pliegue", "semilla"], columns="variante", values="macro")
    piv_l = df.pivot_table(index=["pliegue", "semilla"], columns="variante", values="leve")
    res = {}
    for v in variantes:
        res[v] = {
            "macro": round(float(piv_m[v].mean()), 4), "leve": round(float(piv_l[v].mean()), 4),
            "d_macro_vs_base": round(float((piv_m[v] - piv_m["base"]).mean()), 4),
            "d_leve_vs_base": round(float((piv_l[v] - piv_l["base"]).mean()), 4),
            "d_macro_vs_full": round(float((piv_m[v] - piv_m["full"]).mean()), 4),
            "d_leve_vs_full": round(float((piv_l[v] - piv_l["full"]).mean()), 4),
            "mejor_que_full_macro": int((piv_m[v] > piv_m["full"]).sum()),
            "mejor_que_full_leve": int((piv_l[v] > piv_l["full"]).sum())}
        print(v, res[v], flush=True)
    (t4.SALIDA / "t19_ablacion_proxies.json").write_text(json.dumps({"resumen": res, "corridas": corridas},
                                                                    indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
