"""Construye los auxiliares del front y los guarda SIN objetos de sklearn (solo arrays y JSON):
auxiliares/<dataset>.npz  -> umbrales de zona de duda, media y escala de la estandarizacion, componentes NCA
auxiliares/<dataset>.json -> clases, columnas, parametros del modelo GB_* (se reajusta al cargar la app)
Uso: python construir_auxiliares.py ds1_ingreso_solicitudes ds5_pagos_multimes ...
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PROGRAMA = Path(__file__).resolve().parent
BI = PROGRAMA.parents[1]
V2 = BI / "datasets_multiclase_v2" / "salida"
V3 = BI / "datasets_multiclase_v3"
V4 = BI / "datasets_multiclase_v4"
AUX = PROGRAMA / "auxiliares"
SEMILLA = 42
COBERTURA = 0.70
ORDENADOS = ("ds1", "ds4", "ds5", "ds6", "ds7", "ds8")


def umbrales(pv, K):
    cv, mv = pv.argmax(1), pv.max(1)
    return np.array([np.quantile(mv[cv == k], 1 - COBERTURA) if (cv == k).sum() >= 20 else 0.0 for k in range(K)])


def construir(ds):
    orig, v3, v4 = V2 / ds, V3 / ds, V4 / ds
    par = json.loads((v4 / "parametros_estandarizacion.json").read_text(encoding="utf-8"))
    cols_red = par["columnas_base"]
    gb_cols = [c for c in cols_red if c.startswith("GB_")]
    usa_gb = len(gb_cols) > 0
    tiene_nca = json.loads((v4 / "resumen_v4.json").read_text(encoding="utf-8"))["nca"] is not None
    Xtr = pd.read_csv(orig / "X_train_onehot.csv", float_precision="round_trip").fillna(0)
    Ytr = pd.read_csv(orig / "y_train.csv")["Y"].to_numpy()
    clases = json.loads((orig / "resumen.json").read_text(encoding="utf-8"))["clases"]
    K = len(clases)
    arrays = {}
    meta = {"dataset": ds, "clases": clases, "cols_onehot": list(Xtr.columns), "cols_gb": gb_cols,
            "cols_red": cols_red + ([f"NCA_{i + 1}" for i in range(10)] if tiene_nca else []),
            "ordinal": ds[:3] in ORDENADOS, "usa_gb": usa_gb, "tiene_nca": tiene_nca}
    Xaug = Xtr.to_numpy(float)
    if usa_gb:
        r3 = json.loads((v3 / "resumen_v3.json").read_text(encoding="utf-8"))
        meta["gb_params"] = r3["hiperparametros_elegidos"]
        meta["gb_iter"] = int(r3["iteraciones"])
        gbm = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEMILLA, max_iter=meta["gb_iter"],
                                             early_stopping=False, **meta["gb_params"]).fit(Xtr, Ytr)
        Xva = pd.read_csv(orig / "X_val_onehot.csv", float_precision="round_trip").fillna(0)
        arrays["tau"] = umbrales(gbm.predict_proba(Xva), K)
        gb_tr = pd.read_csv(v3 / "X_train_onehot_gb.csv", float_precision="round_trip")[gb_cols].to_numpy(float)
        Xaug = np.hstack([Xaug, gb_tr])
    sc = StandardScaler().fit(Xaug)
    arrays["media"], arrays["escala"] = sc.mean_, sc.scale_
    print(ds, "media dif:", float(np.max(np.abs(sc.mean_ - np.array(par["media"])))), "| desv dif:", float(np.max(np.abs(sc.scale_ - np.array(par["desviacion"])))))
    if tiene_nca:
        S = sc.transform(Xaug)
        sub = np.random.default_rng(SEMILLA).choice(len(S), size=min(5000, len(S)), replace=False)
        nca = NeighborhoodComponentsAnalysis(n_components=10, random_state=SEMILLA, max_iter=60).fit(S[sub], Ytr[sub])
        arrays["nca"] = nca.components_
    AUX.mkdir(exist_ok=True)
    np.savez(AUX / f"{ds}.npz", **arrays)
    (AUX / f"{ds}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("guardado", ds, "| GB:", usa_gb, "| NCA:", tiene_nca)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        construir(ds)
