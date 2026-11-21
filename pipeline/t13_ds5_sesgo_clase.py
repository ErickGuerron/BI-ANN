"""T13 (ds5, step b): per-class logit bias on the trained sqrt-weighted CE network (multiplier 1.0).

Decision rule: argmax(log p + b), b_AL_DIA = 0. b is fitted on validation by coordinate search maximising
macro F1. Test is read once per seed. Two honesty checks:
  - 2-fold cross-fit inside validation (fit on one half, score the other) to measure how much of the
    validation gain is overfit to the fitting set.
  - paired bootstrap on test of the macro F1 and MORA_LEVE F1 difference (biased minus plain).
Writes resultados_keras/t13_ds5_sesgo_clase.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402

GRID = np.round(np.arange(-1.5, 1.5001, 0.05), 3)
RONDAS = 3
BOOT = 1000


def macro(y, pred, K):
    return f1_score(y, pred, average="macro", labels=range(K), zero_division=0)


def aplicar(P, b):
    return (np.log(np.clip(P, 1e-9, None)) + b).argmax(1)


def ajustar(P, y, K):
    b = np.zeros(K)
    mejor = macro(y, aplicar(P, b), K)
    for _ in range(RONDAS):
        for k in range(1, K):
            for v in GRID:
                cand = b.copy()
                cand[k] = v
                s = macro(y, aplicar(P, cand), K)
                if s > mejor + 1e-9:
                    mejor, b = s, cand
    return b, mejor


def f1_leve(y, pred, K, idx):
    return f1_score(y, pred, average=None, labels=range(K), zero_division=0)[idx]


def main():
    X, y = t4.cargar_datos(t4.DS5)
    pesos, cfg = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    K, idx = len(clases), clases.index("MORA_LEVE")
    w = t4.pesos_sqrt_balanceados(y["train"], K)
    rng = np.random.default_rng(0)
    filas = []
    for semilla in t4.SEMILLAS:
        m = t4.entrenar(X["train"], y["train"], X["val"], y["val"], K, cfg["alpha"], "wce", semilla, pesos_clase=w)
        Pv, Pt = t4.probas(m, X["val"]), t4.probas(m, X["test"])
        b, val_sesgo = ajustar(Pv, y["val"], K)
        val_base = macro(y["val"], Pv.argmax(1), K)

        # 2-fold cross-fit inside validation
        perm = rng.permutation(len(y["val"]))
        mitades = np.array_split(perm, 2)
        ganancias = []
        for a, c in ((0, 1), (1, 0)):
            ia, ic = mitades[a], mitades[c]
            ba, _ = ajustar(Pv[ia], y["val"][ia], K)
            ganancias.append(macro(y["val"][ic], aplicar(Pv[ic], ba), K) - macro(y["val"][ic], Pv[ic].argmax(1), K))

        yt = y["test"]
        p0, p1 = Pt.argmax(1), aplicar(Pt, b)
        d_macro, d_leve = [], []
        for _ in range(BOOT):
            s = rng.integers(0, len(yt), len(yt))
            d_macro.append(macro(yt[s], p1[s], K) - macro(yt[s], p0[s], K))
            d_leve.append(f1_leve(yt[s], p1[s], K, idx) - f1_leve(yt[s], p0[s], K, idx))
        f1c0 = f1_score(yt, p0, average=None, labels=range(K), zero_division=0)
        f1c1 = f1_score(yt, p1, average=None, labels=range(K), zero_division=0)
        fila = {
            "semilla": semilla, "sesgo": [round(float(v), 3) for v in b],
            "val_macro_base": round(val_base, 4), "val_macro_sesgo": round(val_sesgo, 4),
            "ganancia_val_cruzada": [round(g, 4) for g in ganancias],
            "prueba_macro_base": round(macro(yt, p0, K), 4), "prueba_macro_sesgo": round(macro(yt, p1, K), 4),
            "prueba_f1_clase_base": [round(float(v), 4) for v in f1c0],
            "prueba_f1_clase_sesgo": [round(float(v), 4) for v in f1c1],
            "boot_macro_ic95": [round(float(np.percentile(d_macro, 2.5)), 4), round(float(np.percentile(d_macro, 97.5)), 4)],
            "boot_leve_ic95": [round(float(np.percentile(d_leve, 2.5)), 4), round(float(np.percentile(d_leve, 97.5)), 4)],
        }
        filas.append(fila)
        print(json.dumps(fila), flush=True)

    ruta = t4.SALIDA / "t13_ds5_sesgo_clase.json"
    ruta.write_text(json.dumps({"clases": clases, "corridas": filas}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
