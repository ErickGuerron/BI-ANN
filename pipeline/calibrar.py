"""Calibración de las probabilidades de la red (temperature scaling) con comparación contra regresión isotónica.

Proceso:
  1. Se obtienen los logits de la red (salida antes del softmax) para validación y prueba.
  2. Temperature scaling: se busca una sola temperatura T > 0 que minimice la pérdida logarítmica (NLL) en
     VALIDACIÓN, con softmax(z / T). T > 1 suaviza la confianza; T < 1 la afila. No cambia la clase predicha,
     porque dividir entre T no altera el orden de los logits.
  3. Regresión isotónica (comparación, no se usa en la app): un ajuste monótono por clase, uno contra el resto,
     entrenado en VALIDACIÓN. Luego las probabilidades se renormalizan. Puede cambiar la clase predicha.
  4. Prueba: se mide NLL, Brier, ECE (error de calibración esperado) y exactitud en prueba, que no se usa para
     ajustar nada. Solo T se guarda para la app.

Uso:  python pipeline/calibrar.py ds7_historico_garantias ds1_ingreso_solicitudes ds6_gasto_provisiones ds5_pagos_multimes
Salida: auxiliares/calibracion_<dataset>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import inferencia as I  # noqa: E402

TOP4 = ["ds7_historico_garantias", "ds1_ingreso_solicitudes", "ds6_gasto_provisiones", "ds5_pagos_multimes"]
# Datasets donde se aplica temperature scaling. ds6 y ds5 se excluyeron: la temperatura ajustada en validación
# empeoró (o no mejoró) las métricas en prueba. La decisión se tomó después de ver prueba; ver docs/calibracion.md.
CON_TEMPERATURA = ["ds7_historico_garantias", "ds1_ingreso_solicitudes"]


def logits(pesos, X):
    a = X
    capas = len(pesos["coefs"])
    for i in range(capas):
        z = a @ np.array(pesos["coefs"][i]) + np.array(pesos["intercepts"][i])
        a = np.maximum(z, 0) if i < capas - 1 else z
    return a


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def ece(p, y, n_bins=10):
    """Error de calibración esperado: promedio ponderado de |exactitud - confianza| por bins de confianza."""
    conf = p.max(axis=1)
    ok = p.argmax(axis=1) == y
    bordes = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def ajustar_temperatura(z, y):
    r = minimize_scalar(lambda T: nll(softmax(z / T), y), bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def isotonica(pv, yv, pt):
    out = np.zeros_like(pt)
    for k in range(pv.shape[1]):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pv[:, k], (yv == k).astype(float))
        out[:, k] = ir.predict(pt[:, k])
    out = np.clip(out, 1e-6, None)
    return out / out.sum(axis=1, keepdims=True)


def metricas(p, y):
    return {"nll": round(nll(p, y), 4), "brier": round(brier(p, y), 4), "ece": round(ece(p, y), 4),
            "exactitud": round(float(np.mean(p.argmax(axis=1) == y)), 4),
            "exactitud_balanceada": round(float(balanced_accuracy_score(y, p.argmax(axis=1))), 4),
            "confianza_media": round(float(p.max(axis=1).mean()), 4)}


def calibrar(ds):
    v4 = I.BI / "datasets_multiclase_v4" / ds
    v2 = I.V2_SALIDA / ds
    pesos = json.loads((I.PESOS / f"pesos_{ds}.json").read_text(encoding="utf-8"))
    Xv = pd.read_csv(v4 / "X_val_nn.csv").to_numpy(float)
    Xt = pd.read_csv(v4 / "X_test_nn.csv").to_numpy(float)
    yv = pd.read_csv(v2 / "y_val.csv")["Y"].to_numpy().astype(int)
    yt = pd.read_csv(v2 / "y_test.csv")["Y"].to_numpy().astype(int)
    assert Xv.shape[1] == pesos["coefs"][0].__len__(), "la entrada no coincide con la red"

    zv, zt = logits(pesos, Xv), logits(pesos, Xt)
    T = ajustar_temperatura(zv, yv)
    pv, pt = softmax(zv), softmax(zt)
    res = {
        "sin_calibrar": metricas(pt, yt),
        "temperature_scaling": metricas(softmax(zt / T), yt),
        "isotonica": metricas(isotonica(pv, yv, pt), yt),
    }
    salida = {"dataset": ds, "metodo": "temperature_scaling", "temperatura": round(T, 4),
              "ajustada_en": "validacion", "evaluada_en": "prueba", "metricas_prueba": res}
    (I.AUX / f"calibracion_{ds}.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    return salida


if __name__ == "__main__":
    for ds in sys.argv[1:] or CON_TEMPERATURA:
        s = calibrar(ds)
        print(ds, "T =", s["temperatura"])
        for nombre, m in s["metricas_prueba"].items():
            print(f"  {nombre:20} NLL {m['nll']}  Brier {m['brier']}  ECE {m['ece']}  "
                  f"exact {m['exactitud']}  bal {m['exactitud_balanceada']}  conf {m['confianza_media']}")
