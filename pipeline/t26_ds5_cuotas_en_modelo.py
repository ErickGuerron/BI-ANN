"""T26 (ds5, step 3): does the installment-lag signal (t25) add to HGB on the temporal folds?

Same folds, HGB settings and data order as t24. Variants (all HGB on the 101 one-hot columns):
  base        : no proxies
  par         : + dia_desv, dia_ult_menos_med (+ missing flag)
  par+atraso  : par + cuotas_atrasadas_max
  par+atraso+edad : par + cuotas_atrasadas_max + meses_desde_desemb (isolates plain loan age)
No tuning; test read once per fold. Writes resultados_keras/t26_cuotas_en_modelo.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402
import t18_ds5_gb_por_pliegue as t18  # noqa: E402
import t25_ds5_cuotas_atrasadas as t25  # noqa: E402
from t17_ds5_pliegues_temporales import PLIEGUES  # noqa: E402

PAR = ["dia_desv", "dia_ult_menos_med"]
VARIANTES = {
    "base": [],
    "par": PAR,
    "par+atraso": PAR + ["cuotas_atrasadas_max"],
    "par+atraso+edad": PAR + ["cuotas_atrasadas_max", "meses_desde_desemb"],
}


def main():
    r3 = json.loads((t18.V3 / "resumen_v3.json").read_text(encoding="utf-8"))
    params, n_iter = r3["hiperparametros_elegidos"], int(r3["iteraciones"])
    X, y = t4.cargar_datos(t4.DS5)
    pesos, _ = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    idx = clases.index("MORA_LEVE")
    d, F = t14.construir(estricto=True)
    S = t25.senales(d)
    orden = ["train", "val", "test"]
    pos = np.concatenate([np.where(d.Particion.to_numpy() == p)[0] for p in orden])
    d, F, S = (a.iloc[pos].reset_index(drop=True) for a in (d, F, S))
    yA = np.concatenate([y[p] for p in orden])
    Xo = np.vstack([pd.read_csv(t18.V2 / f"X_{p}_onehot.csv", float_precision="round_trip").fillna(0).to_numpy(float)
                    for p in orden])
    corte = d.Corte.to_numpy()
    extra = pd.concat([F[PAR], F[["n_fechas"]].isna().astype(float).rename(columns={"n_fechas": "falta"}), S], axis=1)

    res = {}
    for nombre, cf in PLIEGUES.items():
        m = {k: np.isin(corte, v) for k, v in cf.items()}
        for var, cols in VARIANTES.items():
            M = Xo
            if cols:
                E = extra[cols + (["falta"] if any(c in PAR for c in cols) else [])]
                E = E.fillna(E[m["train"]].median())
                M = np.hstack([Xo, E.to_numpy(float)])
            h = HistGradientBoostingClassifier(class_weight="balanced", random_state=42, max_iter=n_iter,
                                               early_stopping=False, **params).fit(M[m["train"]], yA[m["train"]])
            pr = h.predict(M[m["test"]])
            f = f1_score(yA[m["test"]], pr, average=None, labels=range(4), zero_division=0)
            cm = confusion_matrix(yA[m["test"]], pr, labels=range(4))
            real = cm[clases.index("MORA_31_90")]
            res[f"{nombre}|{var}"] = {
                "macro_f1": round(float(f.mean()), 4), "f1_leve": round(float(f[idx]), 4),
                "f1_clase": [round(float(v), 4) for v in f],
                "pct_3190_a_leve": round(float(real[idx] / max(real.sum(), 1)), 4)}
            print(nombre, var, res[f"{nombre}|{var}"], flush=True)
    resumen = {}
    for var in VARIANTES:
        ks = [f"{n}|{var}" for n in PLIEGUES]
        resumen[var] = {k: round(float(np.mean([res[x][k] for x in ks])), 4)
                        for k in ("macro_f1", "f1_leve", "pct_3190_a_leve")}
        resumen[var]["macro_por_pliegue"] = [res[x]["macro_f1"] for x in ks]
        print(var, resumen[var], flush=True)
    (t4.SALIDA / "t26_cuotas_en_modelo.json").write_text(json.dumps({"resumen": resumen, "detalle": res}, indent=2),
                                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
