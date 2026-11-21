"""T20: subsets of the ds5 payment-timing proxies, with confusion matrices (GB_* refitted per fold).

Follow-up of t19 (leave-one-out): dia_desv and dia_ult_menos_med carried most of the effect.
Variants (variant B, 3 temporal folds x 5 seeds, same data as t18 gb_refit; missing flag always kept):
  base, full (9 proxies),
  par   : dia_desv + dia_ult_menos_med,
  trio  : par + dia_ult,
  cuatro: trio + frac_cuotas_parciales,
  sin_huecos: full minus gap_max, gap_medio, gap_desv, n_fechas, dias_ult_pago_vs_gap_medio (= cuatro).
`cuatro` and `sin_huecos` are the same set by construction; only `cuatro` is trained.
Confusion counts (summed over seeds) are stored per fold and variant so the MORA_31_90 <-> MORA_LEVE
confusion can be read directly. Writes resultados_keras/t20_subconjuntos_proxies.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402
import t18_ds5_gb_por_pliegue as t18  # noqa: E402
from t17_ds5_pliegues_temporales import PLIEGUES, SEMILLAS  # noqa: E402

PAR = ["dia_desv", "dia_ult_menos_med"]
VARIANTES = {
    "base": [],
    "full": None,  # all proxies
    "par": PAR,
    "trio": PAR + ["dia_ult"],
    "cuatro": PAR + ["dia_ult", "frac_cuotas_parciales"],
}


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

    corridas, confusion = [], {}
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
        for var, usar in VARIANTES.items():
            usar = cols if usar is None else usar
            M = XR if not usar else np.hstack([XR, Zs[:, [cols.index(c) for c in usar]], falta])
            conf = np.zeros((K, K), int)
            for semilla in SEMILLAS:
                mod = t4.entrenar(M[m["train"]], yA[m["train"]], M[m["val"]], yA[m["val"]], K, cfg["alpha"], "wce",
                                  semilla, pesos_clase=w)
                P = t4.probas(mod, M[m["test"]])
                r = t4.metricas(yA[m["test"]], P)
                conf += confusion_matrix(yA[m["test"]], P.argmax(1), labels=range(K))
                corridas.append({"pliegue": nombre, "variante": var, "semilla": semilla, "macro_f1": r["macro_f1"],
                                 "f1_clase": r["f1_clase"]})
            confusion[f"{nombre}|{var}"] = conf.tolist()
            print(nombre, var, flush=True)

    df = pd.DataFrame([{"pliegue": c["pliegue"], "variante": c["variante"], "semilla": c["semilla"],
                        "macro": c["macro_f1"], "leve": c["f1_clase"][idx],
                        "f31": c["f1_clase"][clases.index("MORA_31_90")]} for c in corridas])
    pm = df.pivot_table(index=["pliegue", "semilla"], columns="variante", values="macro")
    pl = df.pivot_table(index=["pliegue", "semilla"], columns="variante", values="leve")
    p3 = df.pivot_table(index=["pliegue", "semilla"], columns="variante", values="f31")
    res = {}
    for v in VARIANTES:
        res[v] = {"macro": round(float(pm[v].mean()), 4), "leve": round(float(pl[v].mean()), 4),
                  "f31_90": round(float(p3[v].mean()), 4),
                  "d_macro_vs_base": round(float((pm[v] - pm["base"]).mean()), 4),
                  "d_leve_vs_base": round(float((pl[v] - pl["base"]).mean()), 4),
                  "d_macro_vs_full": round(float((pm[v] - pm["full"]).mean()), 4),
                  "d_leve_vs_full": round(float((pl[v] - pl["full"]).mean()), 4),
                  "mejor_que_full_leve": int((pl[v] > pl["full"]).sum()),
                  "mejor_que_base_leve": int((pl[v] > pl["base"]).sum())}
        print(v, res[v], flush=True)
    (t4.SALIDA / "t20_subconjuntos_proxies.json").write_text(
        json.dumps({"clases": clases, "resumen": res, "confusion": confusion, "corridas": corridas}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
