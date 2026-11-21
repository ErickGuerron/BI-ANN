"""Data sheet of one trained network: weights for the diagram, configuration and test metrics.

Everything is read from files the pipeline already writes (no training, no extra dependencies):
  resultados_convergencia/pesos_<ds>.json               weights, layer sizes, input and class names, configuration
  resultados_convergencia/ajuste_<ds>.json              cross-validation, validation and test metrics
  resultados_convergencia/predicciones_prueba_<ds>.csv  test predictions, used for the confusion matrix
  auxiliares/calibracion_<ds>.json                      temperature, only for the networks where it is applied
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROGRAMA = Path(__file__).resolve().parents[1]
DATOS = PROGRAMA.parent                 # carpeta con ds1_..., ds8_... y sus X_*_nn.csv
PESOS = PROGRAMA / "resultados_convergencia"
LIMITE_BRECHA = 0.05                    # |error de prueba - error de entrenamiento| debe ser menor
LIMITE_BRECHA_ESTRICTO = 0.03           # criterio más estricto que usan docs/ds7_*.md
AUX = PROGRAMA / "auxiliares"

NOTA_ENSAMBLE = ("Se muestra la red base de ds7 (una sola red, sin la garantía como entrada). "
                 "La clasificación de la app usa un ensamble de 5 redes con Valor_Garantiza; "
                 "sus resultados están en docs/ds7_garantia_cascada_monitoreo.md.")
NOTAS = {"ds7_historico_garantias": NOTA_ENSAMBLE}


UMBRAL_INICIAL = 0.2        # el diagrama arranca ocultando las conexiones con menos del 20 % del peso máximo de su capa
ESPACIO_NEURONA = 14        # separación vertical entre entradas en el lienzo (visual_red.py)
PROPORCION_LIENZO = 1.7     # ancho / alto aproximado del lienzo dentro del modal


def separacion_capas(capas):
    """Horizontal distance between layers so that the drawn network has about the canvas proportion.
    Never less than the default 240; a network with 124 inputs is ~1700 tall and would otherwise be a thin strip."""
    huecos = max(len(capas) - 1, 1)
    alto = max(capas) * ESPACIO_NEURONA
    return max(240, int(alto * PROPORCION_LIENZO / huecos))


def _leer(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _propagar(pesos, X):
    """Forward pass of the saved network: ReLU in the hidden layers, softmax at the output."""
    a = X
    for k, (w, b) in enumerate(zip(pesos["coefs"], pesos["intercepts"])):
        a = a @ np.asarray(w) + np.asarray(b)
        if k < len(pesos["coefs"]) - 1:
            a = np.maximum(a, 0.0)
    return _softmax(a)


def evaluar_brecha(error_train, error_test):
    """Train-test gap as defined in the project docs: test error minus training error.

    The limits are checked on the absolute value, so a test error lower than the training error that
    differs by more than the limit also fails."""
    brecha = error_test - error_train
    return {"brecha": brecha, "brecha_abs": abs(brecha),
            "cumple_5": abs(brecha) < LIMITE_BRECHA, "cumple_3": abs(brecha) < LIMITE_BRECHA_ESTRICTO}


def _metricas_particion(P, y, K):
    pred = P.argmax(axis=1)
    recall, f1 = [], []
    for k in range(K):
        tp = float(((pred == k) & (y == k)).sum())
        real, previsto = float((y == k).sum()), float((pred == k).sum())
        r = tp / real if real else 0.0
        p = tp / previsto if previsto else 0.0
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"error": 1.0 - float((pred == y).mean()), "exactitud_balanceada": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "perdida": float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1.0)).mean())}


_CACHE_ENTRENAMIENTO = {}


def metricas_entrenamiento(ds, pesos, historial):
    """Cached per dataset and data folder: evaluating 20 000 rows three times is the slow part of the sheet."""
    clave = (ds, str(DATOS))
    if clave not in _CACHE_ENTRENAMIENTO:
        _CACHE_ENTRENAMIENTO[clave] = _calcular_entrenamiento(ds, pesos, historial)
    return _CACHE_ENTRENAMIENTO[clave]


def _calcular_entrenamiento(ds, pesos, historial):
    """Error, balanced accuracy, macro F1 and log-loss on train, validation and test, plus the gap.

    Computed with the saved weights on the same X_*_nn.csv files the network was trained with. The training
    split is evaluated as it is (without the oversampling used to fit). Returns None if the data are not there."""
    K = len(pesos["clases"])
    por_parte = {}
    try:
        for parte in ("train", "val", "test"):
            X = pd.read_csv(DATOS / ds / f"X_{parte}_nn.csv").to_numpy(dtype=np.float64)
            y = pd.read_csv(DATOS / ds / f"y_{parte}.csv")["Y"].to_numpy(dtype=int)
            por_parte[parte] = _metricas_particion(_propagar(pesos, X), y, K)
    except FileNotFoundError:
        return None
    salida = {m: {p: por_parte[p][m] for p in por_parte} for m in ("error", "exactitud_balanceada", "macro_f1", "perdida")}
    salida.update(evaluar_brecha(por_parte["train"]["error"], por_parte["test"]["error"]))
    salida["brecha_val"] = por_parte["val"]["error"] - por_parte["train"]["error"]
    salida["historial"] = [dict(h) for h in historial]
    return salida


def matriz_confusion(ds, clases):
    """Rows are the real class and columns the predicted class, in the order of `clases`."""
    d = pd.read_csv(PESOS / f"predicciones_prueba_{ds}.csv")
    tabla = pd.crosstab(d["clase_real"], d["clase_predicha"]).reindex(index=clases, columns=clases, fill_value=0)
    return tabla.to_numpy(dtype=int)


def recall_precision(cm):
    """Per-class recall (row-wise) and precision (column-wise); NaN when a class never appears."""
    cm = np.asarray(cm, dtype=float)
    diag = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        return diag / cm.sum(axis=1), diag / cm.sum(axis=0)


def cargar_ficha(ds):
    pesos = _leer(PESOS / f"pesos_{ds}.json")
    ajuste = _leer(PESOS / f"ajuste_{ds}.json")
    clases = list(pesos["clases"])
    cm = matriz_confusion(ds, clases)
    recall, precision = recall_precision(cm)
    ruta_cal = AUX / f"calibracion_{ds}.json"
    return {
        "ds": ds,
        "capas": list(pesos["capas"]),
        "coefs": pesos["coefs"],
        "entradas": list(pesos["entradas"]),
        "clases": clases,
        "config": dict(ajuste["config_elegida"]),
        "metricas": {
            "prueba": ajuste["prueba"],
            "validacion": ajuste["validacion"],
            "cv_promedio": ajuste["cv_promedio"],
            "cv_por_semilla": ajuste["cv_por_semilla"],
            "epoca_final": ajuste["epoca_final"],
            "epocas_entrenadas": ajuste["epocas_entrenadas"],
        },
        "entrenamiento": metricas_entrenamiento(ds, pesos, ajuste["historial"]),
        "confusion": cm.tolist(),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "temperatura": _leer(ruta_cal)["temperatura"] if ruta_cal.exists() else None,
        "nota": NOTAS.get(ds, ""),
    }
