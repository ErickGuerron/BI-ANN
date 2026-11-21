"""Abstention policy for the ds7 ensemble. Pure NumPy/scikit-learn functions, no TensorFlow.

A case is abstained (REVISION_MANUAL) when the normalized entropy of the averaged probabilities is above
umbral_h, or when the largest probability is below umbral_p (0.65 by default).

Threshold choice: maximize the macro F1 of the accepted cases, subject to a minimum coverage for EVERY class
(fraction of each true class that is accepted). Without the per-class constraint the threshold can abstain
almost every minority-class case and still look accurate on the cases it keeps.
"""
import numpy as np
from sklearn.metrics import f1_score

UMBRAL_P = 0.65


def entropia_normalizada(P):
    K = P.shape[1]
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(1) / np.log(K)


def abstencion(P, umbral_h, umbral_p=UMBRAL_P, rechazo_previo=None):
    """rechazo_previo: optional boolean mask of cases already rejected by an earlier filter (out of range)."""
    abst = (entropia_normalizada(P) > umbral_h) | (P.max(1) < umbral_p)
    if rechazo_previo is not None:
        abst = abst | np.asarray(rechazo_previo, dtype=bool)
    return abst


def cobertura_por_clase(y, abst, K):
    return np.array([(~abst[y == k]).mean() if (y == k).any() else np.nan for k in range(K)])


def metricas(y, P, abst, K):
    acepta = ~abst
    pred = P.argmax(1)
    res = {
        "cobertura": round(float(acepta.mean()), 4),
        "cobertura_por_clase": [round(float(v), 4) for v in cobertura_por_clase(y, abst, K)],
        "macro_f1_aceptados": None,
        "exactitud_aceptados": None,
    }
    if acepta.any():
        res["macro_f1_aceptados"] = round(float(f1_score(y[acepta], pred[acepta], average="macro",
                                                         labels=range(K), zero_division=0)), 4)
        res["exactitud_aceptados"] = round(float((y[acepta] == pred[acepta]).mean()), 4)
    fallos = np.where(acepta, pred, -1)
    res["macro_f1_abstenciones_como_error"] = round(float(f1_score(y, fallos, average="macro",
                                                                   labels=range(K), zero_division=0)), 4)
    return res


def elegir_umbral(Pv, yv, K, cobertura_min_clase, grilla, rechazo_previo=None):
    """Return (umbral_h, metricas_val) with the best accepted macro F1 under the per-class coverage floor.
    Returns (None, None) when no threshold in the grid satisfies the floor."""
    mejor = None
    for u in grilla:
        m = metricas(yv, Pv, abstencion(Pv, u, rechazo_previo=rechazo_previo), K)
        cob = np.array(m["cobertura_por_clase"], dtype=float)
        if np.isnan(cob).any() or cob.min() < cobertura_min_clase or m["macro_f1_aceptados"] is None:
            continue
        if mejor is None or m["macro_f1_aceptados"] > mejor[1]["macro_f1_aceptados"]:
            mejor = (float(u), m)
    return mejor if mejor else (None, None)
