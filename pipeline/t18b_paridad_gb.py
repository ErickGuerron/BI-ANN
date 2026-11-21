"""T18b: where does the refit GB_* differ from the stored GB_* on the original partition (F3)?

Reports, per partition and per GB column, the mean absolute difference and the correlation in raw
(unstandardised) space, plus the argmax agreement of GB_P_* and the share of rows that differ. Writes
resultados_keras/t18b_paridad_gb.json.
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


def main():
    r3 = json.loads((t18.V3 / "resumen_v3.json").read_text(encoding="utf-8"))
    params, n_iter = r3["hiperparametros_elegidos"], int(r3["iteraciones"])
    X, y = t4.cargar_datos(t4.DS5)
    d = pd.read_csv(t14.LIMPIO, usecols=["Corte", "Particion"])
    orden = ["train", "val", "test"]
    pos = np.concatenate([np.where(d.Particion.to_numpy() == p)[0] for p in orden])
    d = d.iloc[pos].reset_index(drop=True)
    XA = np.vstack([X[p] for p in orden])
    yA = np.concatenate([y[p] for p in orden])
    Xo = np.vstack([pd.read_csv(t18.V2 / f"X_{p}_onehot.csv", float_precision="round_trip").fillna(0).to_numpy(float)
                    for p in orden])
    corte = d.Corte.to_numpy()
    m = {p: (d.Particion.to_numpy() == p) for p in orden}
    gb = t18.gb_pliegue(Xo, yA, corte, m, params, t18_iter(n_iter), 4)

    # un-standardise the stored GB columns with the stored-GB statistics of the original train
    stored_raw = pd.read_csv(t18.V3 / "X_train_onehot_gb.csv", float_precision="round_trip")[t18.GB_COLS].to_numpy(float)
    sc = StandardScaler().fit(np.hstack([Xo[m["train"]], stored_raw]))
    mu, sd = sc.mean_[Xo.shape[1]:], sc.scale_[Xo.shape[1]:]
    out = {}
    for p in orden:
        guardado = XA[m[p]][:, Xo.shape[1]:] * sd + mu
        nuevo = gb[p]
        dif = np.abs(guardado - nuevo)
        out[p] = {
            "media_abs_por_columna": {c: round(float(dif[:, i].mean()), 5) for i, c in enumerate(t18.GB_COLS)},
            "max_abs_por_columna": {c: round(float(dif[:, i].max()), 4) for i, c in enumerate(t18.GB_COLS)},
            "correlacion": {c: round(float(np.corrcoef(guardado[:, i], nuevo[:, i])[0, 1]), 5)
                            for i, c in enumerate(t18.GB_COLS)},
            "argmax_igual": round(float((guardado[:, :4].argmax(1) == nuevo[:, :4].argmax(1)).mean()), 4),
            "filas_con_alguna_dif_mayor_1e-3": round(float((dif.max(1) > 1e-3).mean()), 4)}
        print(p, "argmax_igual", out[p]["argmax_igual"], "filas_dif", out[p]["filas_con_alguna_dif_mayor_1e-3"],
              "media_abs", out[p]["media_abs_por_columna"], flush=True)
    (t4.SALIDA / "t18b_paridad_gb.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


def t18_iter(n):
    return n


if __name__ == "__main__":
    sys.exit(main())
