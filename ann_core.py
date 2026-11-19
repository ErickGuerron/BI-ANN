"""Núcleo de la red neuronal (MLP de scikit-learn) para los datasets multiclase v4.

Este módulo no decide nada por sí mismo: ofrece las piezas que usan
ajuste.py (búsqueda de hiperparámetros) y, después, la app de Streamlit.
"""
import copy
import os

# Limita los hilos de BLAS para no saturar el equipo mientras se entrena.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

CARPETA_DATOS = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"

DATASETS = [
    "ds1_ingreso_solicitudes",
    "ds2_ciclo_solicitud",
    "ds3_desembolsos",
    "ds4_cierre_cartera_mensual",
    "ds5_pagos_multimes",
    "ds6_gasto_provisiones",
    "ds7_historico_garantias",
    "ds8_notificaciones_cobranza",
]


@dataclass
class ConfigRed:
    capas_ocultas: tuple
    alpha: float = 1e-3            # regularización L2
    tamano_lote: int = 64
    tasa_aprendizaje: float = 1e-3
    sobremuestreo: bool = False    # duplica filas de clases pequeñas, solo en train
    semilla: int = 42

    def a_dict(self):
        d = asdict(self)
        d["capas_ocultas"] = list(self.capas_ocultas)
        return d


def tamano_lote_sugerido(n_train):
    """Regla de batch size según el número de filas de train (equipo con RAM de sobra, CPU limitada)."""
    if n_train < 7000:
        return 32
    if n_train < 13000:
        return 64
    return 128


def receta_original(nombre):
    return json.loads((CARPETA_DATOS / nombre / "receta_red_v4.json").read_text(encoding="utf-8"))


def cargar_dataset(nombre):
    carpeta = CARPETA_DATOS / nombre
    etiquetas = pd.read_csv(carpeta / "y_train.csv")[["Y", "Etiqueta"]].drop_duplicates()
    clases = etiquetas.sort_values("Y")["Etiqueta"].tolist()
    datos = {"nombre": nombre, "clases": clases}
    for parte in ("train", "val", "test"):
        X = pd.read_csv(carpeta / f"X_{parte}_nn.csv").to_numpy(dtype=np.float64)
        y = pd.read_csv(carpeta / f"y_{parte}.csv")["Y"].to_numpy(dtype=int)
        ids = pd.read_csv(carpeta / f"ids_{parte}.csv").iloc[:, 0].to_numpy()
        datos[parte] = {"X": X, "y": y, "ids": ids}
    return datos


def sobremuestrear(X, y, semilla):
    """Duplica al azar filas de las clases pequeñas hasta igualar a la mayor. Solo se usa en train."""
    rng = np.random.default_rng(semilla)
    clases, conteos = np.unique(y, return_counts=True)
    maximo = conteos.max()
    indices = []
    for clase, conteo in zip(clases, conteos):
        idx = np.flatnonzero(y == clase)
        faltan = maximo - conteo
        if faltan > 0:
            idx = np.concatenate([idx, rng.choice(idx, size=faltan, replace=True)])
        indices.append(idx)
    todos = rng.permutation(np.concatenate(indices))
    return X[todos], y[todos]


def preparar_train(config, X, y):
    if config.sobremuestreo:
        return sobremuestrear(X, y, config.semilla)
    return X, y


def crear_modelo(config):
    return MLPClassifier(
        hidden_layer_sizes=tuple(config.capas_ocultas),
        activation="relu",
        solver="adam",
        alpha=config.alpha,
        batch_size=config.tamano_lote,
        learning_rate_init=config.tasa_aprendizaje,
        random_state=config.semilla,
    )


def evaluar(modelo, X, y):
    proba = modelo.predict_proba(X)
    pred = modelo.classes_[proba.argmax(axis=1)]
    return {
        "exactitud_balanceada": float(balanced_accuracy_score(y, pred)),
        "exactitud": float((pred == y).mean()),
        "auc_macro_ovr": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
    }


def entrenar(config, train, val, epocas):
    """Entrena `epocas` épocas completas y guarda el historial de cada una (para la gráfica en Streamlit)."""
    X, y = preparar_train(config, train["X"], train["y"])
    clases = np.unique(train["y"])
    modelo = crear_modelo(config)
    historial = []
    for epoca in range(1, epocas + 1):
        modelo.partial_fit(X, y, classes=clases)
        pred_val = modelo.predict(val["X"])
        historial.append({
            "epoca": epoca,
            "perdida_train": float(modelo.loss_),
            "exactitud_balanceada_val": float(balanced_accuracy_score(val["y"], pred_val)),
        })
    return modelo, historial


def guardar_predicciones(modelo, parte, clases, ruta):
    """CSV con ID, clase real, clase predicha y probabilidad de cada clase, para revisar caso por caso."""
    proba = modelo.predict_proba(parte["X"])
    pred = modelo.classes_[proba.argmax(axis=1)]
    df = pd.DataFrame({
        "id": parte["ids"],
        "clase_real": [clases[i] for i in parte["y"]],
        "clase_predicha": [clases[i] for i in pred],
    })
    df["acierto"] = parte["y"] == pred
    for j, nombre in enumerate(clases):
        df[f"prob_{nombre}"] = proba[:, j]
    df.to_csv(ruta, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------------------
# Parada temprana por validación: las épocas no son fijas. Se entrena hasta que
# la exactitud balanceada de validación deja de mejorar y se conserva la mejor época.
# ---------------------------------------------------------------------------
TOL_VALIDACION = 1e-3   # mejora mínima de la exactitud balanceada para contar como progreso
PACIENCIA = 10          # épocas seguidas sin mejora antes de detener
MAX_EPOCAS = 200        # tope de seguridad


def epoca_de_plateau(perdidas, tol=TOL_VALIDACION, paciencia=PACIENCIA):
    """Primera época que cierra `paciencia` épocas seguidas sin mejora mayor que `tol` (minimizando)."""
    mejor = np.inf
    sin_progreso = 0
    for i, valor in enumerate(perdidas):
        if valor < mejor - tol:
            mejor = valor
            sin_progreso = 0
        else:
            sin_progreso += 1
            if sin_progreso == paciencia:
                return i + 1
    return None


def epoca_optima(score_curva, tol=TOL_VALIDACION, paciencia=PACIENCIA):
    """Devuelve (época con el mejor score, época en que se detiene la parada temprana)."""
    parada = epoca_de_plateau([-s for s in score_curva], tol=tol, paciencia=paciencia) or len(score_curva)
    mejor = int(np.argmax(score_curva[:parada])) + 1
    return mejor, parada


def entrenar_con_parada_temprana(config, train, val, max_epocas=MAX_EPOCAS):
    """Entrena época a época. Devuelve el modelo de la mejor época de validación, su número y el historial."""
    X, y = preparar_train(config, train["X"], train["y"])
    clases = np.unique(train["y"])
    modelo = crear_modelo(config)
    historial = []
    mejor_score, mejor_epoca, mejor_modelo = -np.inf, 0, None
    sin_progreso = 0
    for epoca in range(1, max_epocas + 1):
        modelo.partial_fit(X, y, classes=clases)
        score = float(balanced_accuracy_score(val["y"], modelo.predict(val["X"])))
        historial.append({"epoca": epoca, "perdida_train": float(modelo.loss_), "exactitud_balanceada_val": score})
        if score > mejor_score + TOL_VALIDACION:
            mejor_score, mejor_epoca = score, epoca
            mejor_modelo = copy.deepcopy(modelo)
            sin_progreso = 0
        else:
            sin_progreso += 1
        if sin_progreso == PACIENCIA:
            break
    return mejor_modelo, historial, mejor_epoca
