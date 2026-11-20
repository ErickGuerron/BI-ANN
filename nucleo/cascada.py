"""Two-step decision cascade for ds7 used by the app. NumPy and scikit-learn only: no TensorFlow at runtime.

Step 1 (geometric filter): mean distance to the k nearest training rows. Above d_max the case is rejected as
FUERA_DE_RANGO before the softmax is read.
Step 2 (ensemble ambiguity): the 5 members are averaged. The case goes to REVISION_MANUAL (AMBIGUEDAD) when the
normalized entropy exceeds umbral_h or the largest probability is below umbral_p.
Otherwise the prediction is AUTOMATICO.

Weights and thresholds come from pipeline/exportar_cascada_ds7.py (auxiliares/cascada_<ds>.npz and .json).
The input is the standardized network input, i.e. the output of inferencia.entrada_red.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROGRAMA = Path(__file__).resolve().parents[1]
AUX = PROGRAMA / "auxiliares"
V4 = PROGRAMA.parent

AUTOMATICO = "AUTOMATICO"
REVISION_MANUAL = "REVISION_MANUAL"
FUERA_DE_RANGO = "FUERA_DE_RANGO"
AMBIGUEDAD = "AMBIGUEDAD"


def disponible(ds):
    return (AUX / f"cascada_{ds}.npz").exists() and (AUX / f"cascada_{ds}.json").exists()


def propagar(X, capas):
    """Forward pass of one member: ReLU hidden layer, softmax output."""
    W0, b0, W1, b1 = capas
    z = np.maximum(X @ W0 + b0, 0.0) @ W1 + b1
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def entropia_normalizada(P):
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(1) / np.log(P.shape[1])


@lru_cache(maxsize=None)
def cargar(ds):
    cfg = json.loads((AUX / f"cascada_{ds}.json").read_text(encoding="utf-8"))
    arr = np.load(AUX / f"cascada_{ds}.npz")
    miembros = [tuple(arr[f"{n}_{i}"] for n in ("W0", "b0", "W1", "b1")) for i in range(cfg["miembros"])]
    train = pd.read_csv(V4 / ds / "X_train_nn.csv").to_numpy(np.float32)
    nn = NearestNeighbors(n_neighbors=cfg["k_vecinos"]).fit(train)
    return {"cfg": cfg, "miembros": miembros, "nn": nn}


def evaluar(ds, X):
    """X: (n, d) standardized inputs. Returns one dict per row."""
    d = cargar(ds)
    cfg = d["cfg"]
    X = np.asarray(X, dtype=np.float64)
    P = np.mean([propagar(X, m) for m in d["miembros"]], axis=0)
    dist = d["nn"].kneighbors(X.astype(np.float32))[0].mean(1)
    H = entropia_normalizada(P)
    out = []
    for i in range(len(X)):
        if dist[i] > cfg["d_max"]:
            estado, motivo = REVISION_MANUAL, FUERA_DE_RANGO
        elif H[i] > cfg["umbral_h"] or P[i].max() < cfg["umbral_p"]:
            estado, motivo = REVISION_MANUAL, AMBIGUEDAD
        else:
            estado, motivo = AUTOMATICO, None
        out.append({"estado": estado, "motivo": motivo, "probs": P[i].tolist(), "clase_idx": int(P[i].argmax()),
                    "entropia": float(H[i]), "distancia": float(dist[i])})
    return out
