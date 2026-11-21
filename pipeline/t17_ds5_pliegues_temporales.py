"""T17: forward-in-time folds and 5 seeds for the ds5 payment-timing proxies (base vs proxies).

ds5 is split by cut date (Corte), so valid alternative partitions look only forward in time:
  F1: train Feb-May, val Jun, test Jul-Aug
  F2: train Feb-Jun, val Jul, test Aug-Sep
  F3: train Feb-Jul, val Aug, test Sep-Oct (the original split)
Each fold trains the network of variant B (sqrt-weighted CE, same alpha as the app) with and without the
proxies, 5 seeds each. Early stopping uses the fold validation. Proxies are standardised with the fold's train
statistics. The GB_* inputs are the ones already stored in the v4 files (each row's GB value is out of sample
with respect to its own row); they are not refitted per fold, which is a stated limitation.
Reported: per fold and overall paired differences (proxies minus base) on the fold test, macro F1 and MORA_LEVE
F1, over fold x seed pairs. Writes resultados_keras/t17_pliegues.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402

SEMILLAS = [42, 43, 44, 45, 46]
PLIEGUES = {
    "F1": {"train": [20220228, 20220331, 20220430, 20220531], "val": [20220630], "test": [20220731, 20220831]},
    "F2": {"train": [20220228, 20220331, 20220430, 20220531, 20220630], "val": [20220731],
           "test": [20220831, 20220930]},
    "F3": {"train": [20220228, 20220331, 20220430, 20220531, 20220630, 20220731], "val": [20220831],
           "test": [20220930, 20221031]},
}


def main():
    X, y = t4.cargar_datos(t4.DS5)
    pesos, cfg = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    K, idx = len(clases), clases.index("MORA_LEVE")
    d, F = t14.construir(estricto=True)
    orden = ["train", "val", "test"]
    # dataset_limpio rows follow the v4 order inside each partition; rebuild the global arrays in that order
    pos = np.concatenate([np.where(d.Particion.to_numpy() == p)[0] for p in orden])
    d, F = d.iloc[pos].reset_index(drop=True), F.iloc[pos].reset_index(drop=True)
    XA = np.vstack([X[p] for p in orden])
    yA = np.concatenate([y[p] for p in orden])
    corte = d.Corte.to_numpy()
    assert len(XA) == len(d) and (d.Etiqueta.map({c: i for i, c in enumerate(clases)}).to_numpy() == yA).all()
    cols = list(F.columns)

    filas = []
    for nombre, cfgf in PLIEGUES.items():
        m = {k: np.isin(corte, v) for k, v in cfgf.items()}
        Ftr = F[m["train"]]
        med = Ftr.median()
        Fi = F.fillna(med)
        mu, sd = Fi[m["train"]].mean(), Fi[m["train"]].std().replace(0, 1.0)
        falta = F.isna().to_numpy().astype(np.float32)[:, [cols.index("n_fechas")]]
        Z = np.hstack([((Fi - mu) / sd).to_numpy(np.float32), falta])
        XP = np.hstack([XA, Z])
        w = t4.pesos_sqrt_balanceados(yA[m["train"]], K)
        print(nombre, {k: int(v.sum()) for k, v in m.items()}, flush=True)
        for semilla in SEMILLAS:
            r = {"pliegue": nombre, "semilla": semilla}
            for var, M in (("base", XA), ("proxies", XP)):
                mod = t4.entrenar(M[m["train"]], yA[m["train"]], M[m["val"]], yA[m["val"]], K, cfg["alpha"],
                                  "wce", semilla, pesos_clase=w)
                r[var] = {"val": t4.metricas(yA[m["val"]], t4.probas(mod, M[m["val"]])),
                          "prueba": t4.metricas(yA[m["test"]], t4.probas(mod, M[m["test"]]))}
            r["d_macro"] = round(r["proxies"]["prueba"]["macro_f1"] - r["base"]["prueba"]["macro_f1"], 4)
            r["d_leve"] = round(r["proxies"]["prueba"]["f1_clase"][idx] - r["base"]["prueba"]["f1_clase"][idx], 4)
            filas.append(r)
            print(nombre, semilla, "d_macro", r["d_macro"], "d_leve", r["d_leve"], flush=True)

    def resumen(sel):
        dm, dl = np.array([f["d_macro"] for f in sel]), np.array([f["d_leve"] for f in sel])
        return {"n": len(sel), "d_macro_media": round(float(dm.mean()), 4), "d_macro_desv": round(float(dm.std()), 4),
                "d_macro_positivos": int((dm > 0).sum()),
                "d_leve_media": round(float(dl.mean()), 4), "d_leve_desv": round(float(dl.std()), 4),
                "d_leve_positivos": int((dl > 0).sum()),
                "base_macro": round(float(np.mean([f["base"]["prueba"]["macro_f1"] for f in sel])), 4),
                "proxies_macro": round(float(np.mean([f["proxies"]["prueba"]["macro_f1"] for f in sel])), 4),
                "base_leve": round(float(np.mean([f["base"]["prueba"]["f1_clase"][idx] for f in sel])), 4),
                "proxies_leve": round(float(np.mean([f["proxies"]["prueba"]["f1_clase"][idx] for f in sel])), 4)}

    res = {"semillas": SEMILLAS, "pliegues": PLIEGUES,
           "por_pliegue": {n: resumen([f for f in filas if f["pliegue"] == n]) for n in PLIEGUES},
           "global": resumen(filas), "corridas": filas}
    for n, v in res["por_pliegue"].items():
        print(n, v)
    print("global", res["global"])
    (t4.SALIDA / "t17_pliegues.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
