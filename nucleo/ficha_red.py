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
PESOS = PROGRAMA / "resultados_convergencia"
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
        "confusion": cm.tolist(),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "temperatura": _leer(ruta_cal)["temperatura"] if ruta_cal.exists() else None,
        "nota": NOTAS.get(ds, ""),
    }
