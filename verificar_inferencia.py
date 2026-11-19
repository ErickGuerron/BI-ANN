"""Verifica que la cadena de inferencia reproduce la entrada de la red y las probabilidades guardadas."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inferencia as I

DATASETS = sys.argv[1:] or ["ds1_ingreso_solicitudes", "ds5_pagos_multimes", "ds6_gasto_provisiones", "ds7_historico_garantias"]
N = 60
for ds in DATASETS:
    P, M = I.motor()
    cfg = M.CONFIGS[ds]
    raw = pd.read_csv(Path(cfg["dir"]) / cfg["archivo"], float_precision="round_trip", low_memory=False, dtype={cfg["id"]: str})
    V4 = I.BI / "datasets_multiclase_v4" / ds
    # val: entrada de la red
    ids_val = pd.read_csv(V4 / "ids_val.csv", dtype=str).iloc[:N, 0].tolist() if (V4 / "ids_val.csv").exists() else pd.read_csv(I.V2_SALIDA / ds / "ids_val.csv", dtype=str).iloc[:N, 0].tolist()
    fila = raw.drop_duplicates(cfg["id"]).set_index(cfg["id"]).loc[ids_val].reset_index()
    E = I.entrada_red(ds, fila)
    ref = pd.read_csv(V4 / "X_val_nn.csv").iloc[:N]
    dif_red = float(np.max(np.abs(E.to_numpy(float) - ref.to_numpy(float))))
    # test: probabilidades guardadas
    pred_csv = PESOS = I.PESOS / f"predicciones_prueba_{ds}.csv"
    if pred_csv.exists():
        pc = pd.read_csv(pred_csv, dtype={"id": str}).iloc[:N]
        fila_t = raw.drop_duplicates(cfg["id"]).set_index(cfg["id"]).loc[pc["id"].tolist()].reset_index()
        _, pr = I.predecir(ds, fila_t)
        clases = I.cargar(ds)["aux"]["clases"]
        ref_p = pc[[f"prob_{c}" for c in clases]].to_numpy(float)
        dif_p = float(np.max(np.abs(pr - ref_p)))
        acierto = float(np.mean(np.array(clases)[pr.argmax(1)] == pc["clase_predicha"].to_numpy()))
    else:
        dif_p, acierto = float("nan"), float("nan")
    print(f"{ds}: entrada red max_dif={dif_red:.2e} | probs max_dif={dif_p:.2e} | acuerdo clase={acierto:.3f}")
