"""Export the ds7 cascade for the app: member weights (.npz) and thresholds (.json) into auxiliares/.

Thresholds are chosen on validation only: d_max = 99th percentile of the k-NN score; umbral_h = best accepted
macro F1 under a 0.5 coverage floor per class, OOD rejections counted as not covered.
Checks parity: the NumPy cascade must reproduce the Keras ensemble probabilities on validation.

Requires the members saved by t5_abstencion_por_clase.py. Run in the conda env `bi`.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import t4_keras as T4  # noqa: E402
import politica_abstencion as PA  # noqa: E402
import filtro_ood as F  # noqa: E402
import cascada as C  # noqa: E402
from t5_abstencion_por_clase import ensamble  # noqa: E402
from t6_filtro_ood import cargar_miembros, K_VECINOS, PISO  # noqa: E402

DS = T4.DS7
K = 3


def main():
    X, y = T4.cargar_datos(DS)
    modelos = cargar_miembros()

    knn = F.FiltroKNN(K_VECINOS).ajustar(X["train"])
    d_max = F.umbral_distancia(knn.puntaje(X["val"]))
    Pv = ensamble(modelos, X["val"])
    ood_val = F.rechaza(knn.puntaje(X["val"]), d_max)
    umbral_h, m_val = PA.elegir_umbral(Pv, y["val"], K, PISO, T4.GRID_UMBRAL, rechazo_previo=ood_val)
    if umbral_h is None:
        raise RuntimeError("no entropy threshold satisfies the per-class coverage floor")

    C.AUX.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, m in enumerate(modelos):
        W0, b0 = m.layers[0].get_weights()
        W1, b1 = m.layers[1].get_weights()
        arrays.update({f"W0_{i}": W0, f"b0_{i}": b0, f"W1_{i}": W1, f"b1_{i}": b1})
    np.savez(C.AUX / f"cascada_{DS}.npz", **arrays)
    cfg = {"dataset": DS, "miembros": len(modelos), "k_vecinos": K_VECINOS, "percentil_d_max": F.PERCENTIL,
           "d_max": d_max, "umbral_h": umbral_h, "umbral_p": PA.UMBRAL_P, "piso_cobertura_por_clase": PISO,
           "val": m_val}
    (C.AUX / f"cascada_{DS}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Parity: NumPy cascade vs Keras ensemble on validation.
    C.cargar.cache_clear()
    res = C.evaluar(DS, X["val"])
    P_np = np.array([r["probs"] for r in res])
    dif = float(np.abs(P_np - Pv).max())
    estados = np.array([r["estado"] == C.AUTOMATICO for r in res])
    esperado = ~PA.abstencion(Pv, umbral_h, rechazo_previo=ood_val)
    print("d_max", round(d_max, 4), "umbral_h", umbral_h)
    print("max|prob NumPy - Keras|", dif)
    print("decisiones iguales a la politica de referencia", float((estados == esperado).mean()))
    if dif > 1e-4 or not (estados == esperado).all():
        raise RuntimeError("the exported cascade does not match the reference policy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
