"""Data sheet of one trained network: weights for the diagram, configuration and test metrics.

Everything is read from files the pipeline already writes (no training, no extra dependencies):
  resultados_convergencia/pesos_<ds>.json               weights, layer sizes, input and class names, configuration
  resultados_convergencia/ajuste_<ds>.json              cross-validation, validation and test metrics
  resultados_convergencia/predicciones_prueba_<ds>.csv  test predictions, used for the confusion matrix
  auxiliares/calibracion_<ds>.json                      temperature, only for the networks where it is applied
"""
import json
import math
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


PREGUNTA_CARTA = {
    "ds1_ingreso_solicitudes": {
        "pregunta": "¿Qué clientes presentan un perfil adecuado para recibir un nuevo crédito con un nivel de riesgo aceptable?",
        "origen": "Carta de diseño, pregunta 4",
        "como_responde": ("La red recibe los datos de la solicitud y estima qué tan alta es la cuota frente a la capacidad de pago: "
                          "una probabilidad para cada tramo (holgada, moderada, ajustada, supera). Un perfil en holgada o moderada "
                          "es el que se ajusta a un riesgo aceptable; ajustada o supera indican que la cuota compromete demasiado "
                          "la capacidad de pago. Responde con la clase de mayor probabilidad."),
    },
    "ds2_ciclo_solicitud": {
        "pregunta": "¿Qué solicitudes conviene aprobar, y con qué rapidez, para reducir el riesgo de incumplimiento?",
        "origen": "Derivada de la carta de diseño, preguntas 4 y objetivo de precisión en la aprobación",
        "como_responde": ("La red recibe los datos de la solicitud y estima la probabilidad de que sea rechazada, aprobada con "
                          "desembolso en 7 días o menos, o aprobada con desembolso en más de 7 días. La clase de mayor probabilidad "
                          "es la respuesta."),
    },
    "ds3_desembolsos": {
        "pregunta": "¿Qué clientes son los que cumplen con el plazo de su crédito?",
        "origen": "Carta de diseño, pregunta 1",
        "como_responde": ("La red recibe las características del préstamo desembolsado y estima la probabilidad de que esté "
                          "cancelado, al día o en mora. Cancelado y al día son cumplimiento; en mora es incumplimiento."),
    },
    "ds4_cierre_cartera_mensual": {
        "pregunta": "¿Qué clientes son los que cumplen con el plazo de su crédito?",
        "origen": "Carta de diseño, pregunta 1",
        "como_responde": ("La red recibe el estado de la operación al cierre y las variables de pago, y estima en qué tramo de mora "
                          "estará en el cierre siguiente: al día, de 1 a 30 días o más de 30. Al día cumple el plazo; los otros dos "
                          "tramos no lo cumplen."),
    },
    "ds5_pagos_multimes": {
        "pregunta": "¿Qué clientes son los que cumplen con el plazo de su crédito?",
        "origen": "Carta de diseño, pregunta 1; objetivo de severidad del atraso",
        "como_responde": ("La red recibe el historial de pagos hasta el corte (cuotas pagadas, montos cobrados, días desde el último "
                          "pago, saldo) y estima la probabilidad de cada tramo de mora al cierre: al día, de 1 a 30 días, de 31 a 90 "
                          "o más de 90. Al día cumple el plazo; los otros tres tramos no lo cumplen, con severidad creciente. "
                          "La frontera entre al día y de 1 a 30 días es la que la red separa peor."),
    },
    "ds6_gasto_provisiones": {
        "pregunta": ("¿Qué factores influyen en que un cliente cumpla o incumpla con el plazo de pago de su crédito?"),
        "origen": "Carta de diseño, pregunta 3",
        "como_responde": ("La red recibe las variables de la operación (saldo, vencido, provisiones, morosidad, perfil del socio) "
                          "y estima si la calificación de riesgo del crédito mejorará, se mantendrá o empeorará en el mes siguiente. "
                          "Empeorar es la señal de incumplimiento. La red dice qué operaciones se deterioran, no cuánto pesa cada "
                          "factor; esa parte de la pregunta no la responde esta red."),
    },
    "ds7_historico_garantias": {
        "pregunta": ("¿Qué clientes presentan un perfil adecuado para recibir un nuevo crédito con un nivel de riesgo aceptable?"),
        "origen": "Carta de diseño, pregunta 4",
        "como_responde": ("La red base que se muestra recibe el monto entregado y las características del socio y del crédito, "
                          "y estima si el crédito tiene sin garantía, cobertura baja o cobertura alta. Como no recibe el valor de la "
                          "garantía, solo puede adivinar la cobertura a partir del monto: por eso su respuesta es limitada. La "
                          "clasificación de la app usa un ensamble que sí recibe el valor de la garantía "
                          "(docs/ds7_garantia_cascada_monitoreo.md)."),
    },
    "ds8_notificaciones_cobranza": {
        "pregunta": "¿Qué operaciones requieren un escalamiento de cobranza temprana?",
        "origen": "Derivada del proceso de gestión preventiva de cobranza de la carta de diseño; no es una pregunta literal de la carta",
        "como_responde": ("La red recibe las características del crédito (valor, número de cuotas, tasa), el mes y año de la "
                          "notificación y el perfil del socio, y estima el nivel de escalamiento: bajo, medio o alto. Escalamiento "
                          "alto indica operaciones que necesitan una gestión de cobranza más intensa. No usa el historial de pagos."),
    },
}

ALFA_HOEFFDING = 0.05       # nivel de confianza de los márgenes: 95 %
CV_K = 5                    # pliegues de la validación cruzada estratificada de la búsqueda (pipeline/ajuste_convergencia.py, --folds por defecto)
MARGEN_OBJETIVO = 0.02      # ±2 pp: margen de error que se exige a la prueba para la justificación


def margen_hoeffding(n, alfa=ALFA_HOEFFDING):
    """Half-width of the error interval that holds with probability 1 − alfa for a test set of n rows.

    Hoeffding: P(|error_medido − error_real| ≥ ε) ≤ 2·exp(−2·n·ε²), so ε = sqrt(ln(2/α) / (2n)). It assumes
    the rows are independent of each other and of the training, which temporal splits only approximate."""
    return math.sqrt(math.log(2.0 / alfa) / (2.0 * n))


def n_hoeffding(margen, alfa=ALFA_HOEFFDING):
    """Smallest test-set size whose Hoeffding half-width is at most `margen`."""
    return math.ceil(math.log(2.0 / alfa) / (2.0 * margen ** 2))


def dimension_red(pesos, n_train):
    """Number of trainable parameters of the saved network and how many training rows it has per parameter.

    Weights and intercepts of every layer. The rule of thumb of 10 rows per parameter is a heuristic, not a
    bound; it is shown so the gap between the network's size and the data is visible."""
    por_capa = []
    for w, b in zip(pesos["coefs"], pesos["intercepts"]):
        por_capa.append(int(np.asarray(w).size + np.asarray(b).size))
    total = sum(por_capa)
    return {"entradas": len(pesos["entradas"]), "capas": list(pesos["capas"]), "parametros_por_capa": por_capa,
            "parametros": total, "muestras_train": n_train,
            "muestras_por_parametro": (n_train / total) if n_train else None,
            "regla_10_por_parametro": 10 * total}


def contraste_dimension(dimension, n_prueba, n_para_2_pp, n_para_1_pp):
    """Compares the network's real size with the thresholds.

    Rule of 10: 10 training rows per parameter, using the parameters the network really has.
    Hoeffding: test rows needed for ±2 pp and ±1 pp (the same for every network, by construction).
    Both thresholds are checked against the rule-of-10 value, which is the only one the network's size defines."""
    P = dimension["parametros"]
    regla = 10 * P
    ntr = dimension["muestras_train"]
    return {
        "parametros": P,
        "regla_10": regla,
        "train": ntr,
        "train_cumple_regla_10": (ntr >= regla) if ntr is not None else None,
        "regla_10_supera_hoeffding_2_pp": regla >= n_para_2_pp,
        "regla_10_supera_hoeffding_1_pp": regla >= n_para_1_pp,
        "prueba": n_prueba,
        "prueba_cumple_2_pp": n_prueba >= n_para_2_pp,
        "prueba_cumple_1_pp": n_prueba >= n_para_1_pp,
    }


def mcc_multiclase(cm):
    """Matthews correlation coefficient for several classes (Gorodkin's generalisation), from a confusion matrix
    with real classes in rows. 1 is perfect, 0 is chance, and it does not reward predicting the majority class."""
    C = np.asarray(cm, dtype=float)
    s = C.sum()
    c = np.trace(C)
    p = C.sum(axis=0)           # predicted totals
    t = C.sum(axis=1)           # real totals
    numerador = c * s - p @ t
    denominador = math.sqrt((s * s - p @ p) * (s * s - t @ t))
    return float(numerador / denominador) if denominador else 0.0


def brier_multiclase(P, y):
    """Mean squared error between the probability vector and the one-hot real class (0 is perfect)."""
    oh = np.zeros_like(P)
    oh[np.arange(len(y)), y] = 1.0
    return float(((P - oh) ** 2).sum(axis=1).mean())


def ece_top_label(P, y, bins=10):
    """Expected calibration error on the top probability: average |accuracy − confidence| per bin, weighted by size."""
    conf = P.max(axis=1)
    acierto = (P.argmax(axis=1) == y).astype(float)
    bordes = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.sum() / len(y) * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


def _filas_entrenamiento(ds):
    ruta = DATOS / ds / "ids_train.csv"
    return len(pd.read_csv(ruta)) if ruta.exists() else None


def _justificacion(ds, pesos, clases, cm):
    """Test-set margins, network size and extra measures for the justification tab. Reads the saved test
    predictions (probabilities and real class) and the number of training rows; no training."""
    d = pd.read_csv(PESOS / f"predicciones_prueba_{ds}.csv")
    P = d[[f"prob_{c}" for c in clases]].to_numpy(dtype=float)
    y = d["clase_real"].map({c: k for k, c in enumerate(clases)}).to_numpy(dtype=int)
    n = len(d)
    n_train = _filas_entrenamiento(ds)
    n_fold = n_train / CV_K if n_train else None
    return {
        "pregunta": PREGUNTA_CARTA.get(ds, {"pregunta": "", "origen": "", "como_responde": ""}),
        "n_prueba": n,
        "hoeffding": {"alfa": ALFA_HOEFFDING, "margen_error": margen_hoeffding(n),
                      "n_para_2_pp": n_hoeffding(0.02), "n_para_1_pp": n_hoeffding(0.01),
                      "cumple_2_pp": n >= n_hoeffding(MARGEN_OBJETIVO)},
        "validacion_cruzada": {"k": CV_K, "tipo": "estratificada, aleatoria, sobre train",
                               "filas_por_pliegue": n_fold,
                               "margen_por_pliegue": margen_hoeffding(n_fold) if n_fold else None},
        "dimension": dimension_red(pesos, n_train),
        "contraste": contraste_dimension(dimension_red(pesos, n_train), n, n_hoeffding(0.02), n_hoeffding(0.01)),
        "extra": {"mcc": mcc_multiclase(cm), "brier": brier_multiclase(P, y), "ece": ece_top_label(P, y)},
    }


def cargar_ficha(ds):
    pesos = _leer(PESOS / f"pesos_{ds}.json")
    ajuste = _leer(PESOS / f"ajuste_{ds}.json")
    clases = list(pesos["clases"])
    cm = matriz_confusion(ds, clases)
    recall, precision = recall_precision(cm)
    ruta_cal = AUX / f"calibracion_{ds}.json"
    return {
        "justificacion": _justificacion(ds, pesos, clases, cm),
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
