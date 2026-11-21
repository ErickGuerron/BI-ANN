"""Valor_Garantiza as one extra input of the ds7 network (column 97 after the 96 standardized inputs).

The input is log1p(Valor_Garantiza), standardized with the training mean and standard deviation. A missing
or blank amount is 0, the same criterion the training matrix used. Negative amounts are clipped to 0.
No coverage ratio and no Tiene_Garantia flag is added: the network must learn the relation itself.

Parameters are written by pipeline/t8_valor_garantiza.py to auxiliares/garantia_<ds>.json.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

AUX = Path(__file__).resolve().parents[1] / "auxiliares"
CAMPO = "Valor_Garantiza"


def transformar(g):
    g = np.nan_to_num(np.asarray(g, dtype=float), nan=0.0)
    return np.log1p(np.clip(g, 0.0, None))


def _ruta(ds):
    return AUX / f"garantia_{ds}.json"


def disponible(ds):
    return _ruta(ds).exists()


def ajustar(ds, g_train):
    """Compute and save the standardization parameters from the training amounts."""
    t = transformar(g_train)
    params = {"media": float(t.mean()), "desv": float(t.std() + 1e-9)}
    AUX.mkdir(parents=True, exist_ok=True)
    _ruta(ds).write_text(json.dumps(params), encoding="utf-8")
    parametros.cache_clear()
    return params


@lru_cache(maxsize=None)
def parametros(ds):
    return json.loads(_ruta(ds).read_text(encoding="utf-8"))


def columna(ds, g):
    p = parametros(ds)
    return (transformar(g) - p["media"]) / p["desv"]


def agregar(ds, X, g):
    """X: (n, 96) standardized inputs; g: (n,) amounts. Returns (n, 97)."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, columna(ds, g).reshape(-1, 1)])
