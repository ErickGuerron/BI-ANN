"""T4: Keras/TensorFlow versions of the ds5 and ds7 networks.

Architecture, same as the app: Input -> Dense(32, relu) -> Dense(K, softmax), with L2 on the kernels.
L2 is mapped from scikit-learn's convention: sklearn adds 0.5 * alpha * sum(W^2) / n_samples to the loss,
which is 0.5 * alpha / n_samples as Keras kernel_regularizer.L2 on the batch-mean loss.

Steps:
  1. replicate: load the app weights into Keras and compare with the NumPy forward pass.
  2. ds5: CE with oversampling (baseline, same recipe as the app) vs sqrt-weighted CE vs focal loss,
     without oversampling, 3 seeds each. The winner is chosen by validation macro F1 only.
  3. ds7: ensemble of 5 Keras networks; abstention policy tuned on validation with a minimum coverage;
     test is reported once.

Writes resultados_keras/t4_resumen.json. Keras training is not bit-for-bit deterministic on CPU.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import callbacks, layers, optimizers, regularizers  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

PROGRAMA = Path(__file__).resolve().parents[1]
V4 = PROGRAMA.parent
PESOS = PROGRAMA / "resultados_convergencia"
AUX = PROGRAMA / "auxiliares"
SALIDA = PROGRAMA / "resultados_keras"
sys.path.insert(0, str(PROGRAMA / "nucleo"))
import ann_core  # noqa: E402

DS5 = "ds5_pagos_multimes"
DS7 = "ds7_historico_garantias"
LOTE = 128
TASA = 1e-3
MAX_EPOCAS = 300
PACIENCIA = 10
SEMILLAS = [42, 43, 44]
MIEMBROS_ENSAMBLE = [42, 43, 44, 45, 46]
UMBRAL_P = 0.65
COBERTURA_MINIMA = 0.70
GRID_UMBRAL = np.round(np.arange(0.05, 1.0001, 0.01), 4)


def cargar_datos(ds):
    carpeta = V4 / ds
    X = {p: pd.read_csv(carpeta / f"X_{p}_nn.csv").to_numpy(np.float32) for p in ("train", "val", "test")}
    y = {p: pd.read_csv(carpeta / f"y_{p}.csv")["Y"].to_numpy().astype(np.int64) for p in ("train", "val", "test")}
    return X, y


def cargar_config(ds):
    pesos = json.loads((PESOS / f"pesos_{ds}.json").read_text(encoding="utf-8"))
    return pesos, pesos["config"]


def construir(d_entrada, K, alpha, n_entreno):
    """Input -> Dense(32, relu) -> Dense(K, softmax). L2 on kernels, mapped from sklearn's alpha."""
    reg = regularizers.L2(0.5 * alpha / n_entreno)
    return keras.Sequential([
        keras.Input(shape=(d_entrada,)),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dense(K, activation="softmax", kernel_regularizer=reg),
    ])


def pesos_sqrt_balanceados(y, K):
    """Class weights sqrt(N / (K n_k)), scaled so that the weighted mean sample weight is 1."""
    cuenta = np.bincount(y, minlength=K).astype(np.float64)
    w = np.sqrt(len(y) / (K * cuenta))
    return (w * len(y) / w[y].sum()).astype(np.float32)


def focal(gamma, alpha_clase, K):
    """Categorical focal loss with per-class alpha: -alpha_y (1 - p_y)^gamma log p_y."""
    alpha_t = tf.constant(alpha_clase, dtype=tf.float32)

    def perdida(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_y = tf.reduce_sum(tf.one_hot(y, K) * p, axis=-1)
        return -tf.gather(alpha_t, y) * tf.pow(1.0 - p_y, gamma) * tf.math.log(p_y)

    return perdida


def entrenar(Xtr, ytr, Xva, yva, K, alpha, perdida, semilla, gamma=2.0, pesos_clase=None):
    keras.utils.set_random_seed(semilla)
    modelo = construir(Xtr.shape[1], K, alpha, len(Xtr))
    if perdida == "focal":
        funcion = focal(gamma, pesos_clase, K)
    else:
        funcion = "sparse_categorical_crossentropy"
    modelo.compile(optimizer=optimizers.Adam(learning_rate=TASA), loss=funcion)
    parada = callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
    clase_peso = None
    if perdida == "wce":
        clase_peso = {k: float(w) for k, w in enumerate(pesos_clase)}
    modelo.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCAS, batch_size=LOTE, shuffle=True,
               class_weight=clase_peso, callbacks=[parada], verbose=0)
    return modelo


def probas(modelo, X):
    return modelo.predict(X, batch_size=4096, verbose=0)


def metricas(y, P):
    pred = P.argmax(1)
    return {
        "bal": round(float(balanced_accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=range(P.shape[1]), zero_division=0)), 4),
        "f1_clase": [round(float(v), 4) for v in f1_score(y, pred, average=None, labels=range(P.shape[1]),
                                                           zero_division=0)],
    }


def replicar(ds):
    """Load the app weights into Keras and compare the forward pass with the NumPy one."""
    pesos, cfg = cargar_config(ds)
    X, y = cargar_datos(ds)
    K = len(pesos["clases"])
    modelo = construir(X["val"].shape[1], K, 0.0, 1)
    modelo.layers[0].set_weights([np.asarray(pesos["coefs"][0], np.float32),
                                  np.asarray(pesos["intercepts"][0], np.float32)])
    modelo.layers[1].set_weights([np.asarray(pesos["coefs"][1], np.float32),
                                  np.asarray(pesos["intercepts"][1], np.float32)])
    a = X["val"]
    a = np.maximum(a @ np.asarray(pesos["coefs"][0]) + np.asarray(pesos["intercepts"][0]), 0)
    logits = a @ np.asarray(pesos["coefs"][1]) + np.asarray(pesos["intercepts"][1])
    e = np.exp(logits - logits.max(1, keepdims=True))
    numpy_p = e / e.sum(1, keepdims=True)
    keras_p = probas(modelo, X["val"])
    return {
        "ds": ds,
        "config_app": cfg,
        "max_abs_diff_probs": float(np.abs(numpy_p - keras_p).max()),
        "argmax_iguales": round(float((numpy_p.argmax(1) == keras_p.argmax(1)).mean()), 6),
        "bal_val_app_sin_temperatura": metricas(y["val"], keras_p)["bal"],
    }


def ds5():
    X, y = cargar_datos(DS5)
    pesos, cfg = cargar_config(DS5)
    K = len(pesos["clases"])
    alpha = cfg["alpha"]
    resultados = {}
    for nombre, opciones in {
        "A_ce_sobremuestreo": dict(sobre=True, perdida="ce"),
        "B_wce_sqrt": dict(sobre=False, perdida="wce"),
        "C_focal_g2": dict(sobre=False, perdida="focal", gamma=2.0),
    }.items():
        filas = []
        for semilla in SEMILLAS:
            if opciones["sobre"]:
                Xt, yt = ann_core.sobremuestrear(X["train"], y["train"], semilla)
                pesos_clase = None
            else:
                Xt, yt = X["train"], y["train"]
                pesos_clase = pesos_sqrt_balanceados(yt, K)
            modelo = entrenar(Xt, yt, X["val"], y["val"], K, alpha, opciones["perdida"], semilla,
                              gamma=opciones.get("gamma", 2.0), pesos_clase=pesos_clase)
            filas.append({"semilla": semilla,
                          "val": metricas(y["val"], probas(modelo, X["val"])),
                          "prueba": metricas(y["test"], probas(modelo, X["test"]))})
        val_f1 = [f["val"]["macro_f1"] for f in filas]
        resultados[nombre] = {
            "val_macro_f1_media": round(float(np.mean(val_f1)), 4),
            "val_macro_f1_desv": round(float(np.std(val_f1)), 4),
            "corridas": filas,
        }
        print(nombre, resultados[nombre]["val_macro_f1_media"], resultados[nombre]["val_macro_f1_desv"], flush=True)
    elegido = max(resultados, key=lambda k: resultados[k]["val_macro_f1_media"])
    return {"ds": DS5, "alpha_app": alpha, "variantes": resultados, "elegido_por_validacion": elegido}


def entropia_normalizada(P):
    K = P.shape[1]
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(1) / np.log(K)


def politica(P, umbral_h):
    """True = abstain (REVISIÓN_MANUAL): normalized entropy above the threshold, or max probability below 0.65."""
    return (entropia_normalizada(P) > umbral_h) | (P.max(1) < UMBRAL_P)


def ece(P, y, bins=10):
    conf = P.max(1)
    acierto = (P.argmax(1) == y).astype(float)
    total = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return round(float(total), 4)


def metricas_con_abstencion(y, P, abstiene, K):
    pred = P.argmax(1)
    acep = ~abstiene
    res = {"cobertura": round(float(acep.mean()), 4)}
    if acep.any():
        res["macro_f1_aceptados"] = round(float(f1_score(y[acep], pred[acep], average="macro",
                                                         labels=range(K), zero_division=0)), 4)
        res["exactitud_aceptados"] = round(float((y[acep] == pred[acep]).mean()), 4)
    else:
        res["macro_f1_aceptados"] = None
        res["exactitud_aceptados"] = None
    fallos = np.where(acep, pred, -1)
    res["macro_f1_abstenciones_como_error"] = round(float(f1_score(y, fallos, average="macro",
                                                                  labels=range(K), zero_division=0)), 4)
    res["abstencion_por_clase"] = [round(float(abstiene[y == k].mean()), 4) for k in range(K)]
    return res


def ds7():
    X, y = cargar_datos(DS7)
    pesos, cfg = cargar_config(DS7)
    K = len(pesos["clases"])
    alpha = cfg["alpha"]
    miembros_val, miembros_test, individuales = [], [], []
    for semilla in MIEMBROS_ENSAMBLE:
        if cfg["sobremuestreo"]:
            Xt, yt = ann_core.sobremuestrear(X["train"], y["train"], semilla)
        else:
            Xt, yt = X["train"], y["train"]
        modelo = entrenar(Xt, yt, X["val"], y["val"], K, alpha, "ce", semilla)
        pv, pt = probas(modelo, X["val"]), probas(modelo, X["test"])
        miembros_val.append(pv)
        miembros_test.append(pt)
        individuales.append({"semilla": semilla, "val": metricas(y["val"], pv)["macro_f1"],
                             "prueba": metricas(y["test"], pt)["macro_f1"]})
        print("miembro", semilla, individuales[-1], flush=True)

    Pv = np.mean(miembros_val, axis=0)
    Pt = np.mean(miembros_test, axis=0)
    candidatos = []
    for umbral in GRID_UMBRAL:
        abst = politica(Pv, umbral)
        m = metricas_con_abstencion(y["val"], Pv, abst, K)
        candidatos.append({"umbral_H": float(umbral), **m})
    validos = [c for c in candidatos if c["cobertura"] >= COBERTURA_MINIMA and c["macro_f1_aceptados"] is not None]
    if not validos:
        raise RuntimeError("no threshold reaches the minimum coverage on validation")
    elegido = max(validos, key=lambda c: c["macro_f1_aceptados"])
    umbral = elegido["umbral_H"]

    abst_test = politica(Pt, umbral)
    return {
        "ds": DS7,
        "alpha_app": alpha,
        "sobremuestreo_app": cfg["sobremuestreo"],
        "miembros": individuales,
        "ensamble_sin_abstencion": {"val": metricas(y["val"], Pv), "prueba": metricas(y["test"], Pt)},
        "ece_val": ece(Pv, y["val"]),
        "ece_prueba": ece(Pt, y["test"]),
        "umbral_H_elegido": umbral,
        "politica_val": elegido,
        "politica_prueba": metricas_con_abstencion(y["test"], Pt, abst_test, K),
        "barrido_val": candidatos,
    }


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = {"replicacion": [replicar(DS5), replicar(DS7)]}
    print("replicacion", json.dumps(resumen["replicacion"], indent=1), flush=True)
    resumen["ds5"] = ds5()
    resumen["ds7"] = ds7()
    ruta = SALIDA / "t4_resumen.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
