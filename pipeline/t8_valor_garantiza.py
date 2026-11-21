"""T8: ds7 ensemble with Valor_Garantiza as an extra input (option 1, no Tiene_Garantia, no coverage ratio).

1. Build the 97-column matrices: the 96 v4 inputs plus the standardized log1p of Valor_Garantiza.
2. Train the same 5-member Keras ensemble as T4/T5 (hidden 32, same alpha, oversampling, same seeds).
3. Thresholds on validation only: d_max (k-NN, 99th percentile) and the entropy threshold under a 0.5
   coverage floor per class, OOD rejections counted as not covered.
4. Export to auxiliares/ with the _garantia suffix (the base cascade files are kept) and check parity with Keras.

Writes resultados_keras/t8_resumen.json. Run in the conda env `bi`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import ann_core  # noqa: E402
import cascada as C  # noqa: E402
import filtro_ood as F  # noqa: E402
import garantia as G  # noqa: E402
import politica_abstencion as PA  # noqa: E402
import t4_keras as T4  # noqa: E402
from t5_abstencion_por_clase import SALIDA, ensamble  # noqa: E402

DS = T4.DS7
K = 3
PISO = 0.5
K_VECINOS = 5
CARPETA_MIEMBROS = SALIDA / "t8_miembros"
RAW = T4.V4.parent / "para_companero_cadena_datasets" / "datasets_bayes" / "datasets" / f"{DS}.csv"
NOMBRE = f"{DS}_garantia"


def montos():
    raw = pd.read_csv(RAW, dtype={"Num_Operacion": str})
    raw[G.CAMPO] = pd.to_numeric(raw[G.CAMPO], errors="coerce")
    raw = raw.set_index("Num_Operacion")
    out = {}
    for p in ("train", "val", "test"):
        ids = pd.read_csv(T4.V4 / DS / f"ids_{p}.csv", dtype=str)["Num_Operacion"]
        out[p] = raw.loc[ids, G.CAMPO].fillna(0).to_numpy()
    return out


def recall_por_clase(y, P):
    pred = P.argmax(1)
    return [round(float((pred[y == k] == k).mean()), 4) for k in range(K)]


def main():
    X0, y = T4.cargar_datos(DS)
    _, cfg_app = T4.cargar_config(DS)
    g = montos()
    G.ajustar(DS, g["train"])
    X = {p: G.agregar(DS, X0[p], g[p]).astype(np.float32) for p in X0}
    print("entradas", X["train"].shape[1], flush=True)

    CARPETA_MIEMBROS.mkdir(parents=True, exist_ok=True)
    modelos = []
    for semilla in T4.MIEMBROS_ENSAMBLE:
        Xt, yt = ann_core.sobremuestrear(X["train"], y["train"], semilla) if cfg_app["sobremuestreo"] else (X["train"], y["train"])
        m = T4.entrenar(Xt, yt, X["val"], y["val"], K, cfg_app["alpha"], "ce", semilla)
        m.save(CARPETA_MIEMBROS / f"miembro_{semilla}.keras")
        modelos.append(m)
        print("miembro", semilla, T4.metricas(y["val"], T4.probas(m, X["val"]))["macro_f1"], flush=True)

    Pv, Pt = ensamble(modelos, X["val"]), ensamble(modelos, X["test"])
    knn = F.FiltroKNN(K_VECINOS).ajustar(X["train"])
    d_max = F.umbral_distancia(knn.puntaje(X["val"]))
    ood_val = F.rechaza(knn.puntaje(X["val"]), d_max)
    ood_test = F.rechaza(knn.puntaje(X["test"]), d_max)
    umbral_h, m_val = PA.elegir_umbral(Pv, y["val"], K, PISO, T4.GRID_UMBRAL, rechazo_previo=ood_val)
    if umbral_h is None:
        raise RuntimeError("no entropy threshold satisfies the per-class coverage floor")

    resumen = {
        "entradas": int(X["train"].shape[1]), "d_max": d_max, "umbral_H": umbral_h,
        "ensamble_val": {**T4.metricas(y["val"], Pv), "recall": recall_por_clase(y["val"], Pv)},
        "ensamble_prueba": {**T4.metricas(y["test"], Pt), "recall": recall_por_clase(y["test"], Pt)},
        "ece_prueba": T4.ece(Pt, y["test"]),
        "cascada_val": m_val,
        "cascada_prueba": PA.metricas(y["test"], Pt, PA.abstencion(Pt, umbral_h, rechazo_previo=ood_test), K),
        "referencia_sin_garantia": {"macro_f1_prueba": 0.6461, "macro_f1_val": 0.6625},
    }

    arrays = {}
    for i, m in enumerate(modelos):
        W0, b0 = m.layers[0].get_weights()
        W1, b1 = m.layers[1].get_weights()
        arrays.update({f"W0_{i}": W0, f"b0_{i}": b0, f"W1_{i}": W1, f"b1_{i}": b1})
    np.savez(C.AUX / f"cascada_{NOMBRE}.npz", **arrays)
    np.save(C.AUX / f"cascada_{NOMBRE}_train.npy", X["train"])
    cfg = {"dataset": DS, "miembros": len(modelos), "k_vecinos": K_VECINOS, "percentil_d_max": F.PERCENTIL,
           "d_max": d_max, "umbral_h": umbral_h, "umbral_p": PA.UMBRAL_P, "piso_cobertura_por_clase": PISO,
           "usa_garantia": True, "entradas": int(X["train"].shape[1]), "train_npy": f"cascada_{NOMBRE}_train.npy",
           "val": m_val}
    (C.AUX / f"cascada_{NOMBRE}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    C.cargar.cache_clear()
    res = C.evaluar(DS, X["val"])
    dif = float(np.abs(np.array([r["probs"] for r in res]) - Pv).max())
    iguales = float((np.array([r["estado"] == C.AUTOMATICO for r in res])
                     == ~PA.abstencion(Pv, umbral_h, rechazo_previo=ood_val)).mean())
    resumen["paridad"] = {"max_abs_prob": dif, "decisiones_iguales": iguales}
    print("paridad", resumen["paridad"], flush=True)
    if dif > 1e-4 or iguales < 1.0:
        raise RuntimeError("the exported cascade does not match the Keras ensemble")

    ruta = SALIDA / "t8_resumen.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ("ensamble_prueba", "cascada_prueba", "ece_prueba", "umbral_H", "d_max")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
