"""Convierte una fila cruda (como la escribe el usuario en el formulario) en la entrada de la red
y devuelve la clase predicha con sus probabilidades.

Cadena (idéntica a la de entrenamiento):
  1. preparacion.py + preprocesador.joblib (limpieza, derivadas, log1p, one-hot)   -> X one-hot
  2. HistGradientBoosting completo (variables GB_*, solo si el conjunto las usa)   -> X + GB_*
     (se reajusta al cargar, con los datos de train y los hiperparametros guardados)
  3. Estandarizacion con media y escala de train (guardadas en auxiliares/*.npz)  -> estandarizado
  4. NCA de 10 componentes (solo conjuntos N): S @ componentes.T                  -> entrada de la red
  5. Red MLP entrenada (pesos_<dataset>.json): ReLU en ocultas, softmax en salida
"""
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROGRAMA = Path(__file__).resolve().parents[1]
BI = PROGRAMA.parents[1]
PREP = BI / "para_companero_cadena_datasets"
V2_SALIDA = BI / "datasets_multiclase_v2" / "salida"
AUX = PROGRAMA / "auxiliares"

# Money fields accept cents even when every training value was a round number.
PREFIJOS_MONETARIOS = ("Valor_", "Monto_", "Saldo_", "Total_", "Provision_")


def es_monetario(nombre):
    return nombre.startswith(PREFIJOS_MONETARIOS)
PESOS = PROGRAMA / "resultados_convergencia"
SEMILLA = 42


@lru_cache(maxsize=1)
def motor():
    """Carga el motor de preparación (preparacion.py) y la configuración por dataset (preparar_mc.py)."""
    os.environ["PEDRO_BI"] = str(PREP)
    for ruta in (PREP / "datasets_preparados", PREP / "datasets_multiclase_v2"):
        if str(ruta) not in sys.path:
            sys.path.insert(0, str(ruta))
    import preparacion as P
    import preparar_mc as M
    return P, M


def _ajustar_gb(ds, meta):
    """Reajusta el HistGradientBoosting completo con train (mismos hiperparametros y semilla que el entrenamiento)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr = pd.read_csv(V2_SALIDA / ds / "X_train_onehot.csv", float_precision="round_trip").fillna(0)
    ytr = pd.read_csv(V2_SALIDA / ds / "y_train.csv")["Y"].to_numpy()
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEMILLA, max_iter=meta["gb_iter"],
                                          early_stopping=False, **meta["gb_params"]).fit(Xtr, ytr)


@lru_cache(maxsize=None)
def cargar(ds):
    P, M = motor()
    pre = joblib.load(V2_SALIDA / ds / "preprocesador.joblib")
    meta = json.loads((AUX / f"{ds}.json").read_text(encoding="utf-8"))
    arr = np.load(AUX / f"{ds}.npz")
    aux = dict(meta)
    aux["media"], aux["escala"] = arr["media"], arr["escala"]
    aux["tau"] = arr["tau"] if "tau" in arr.files else None
    aux["nca"] = arr["nca"] if "nca" in arr.files else None
    ruta_cal = AUX / f"calibracion_{ds}.json"
    aux["temperatura"] = json.loads(ruta_cal.read_text(encoding="utf-8"))["temperatura"] if ruta_cal.exists() else 1.0
    aux["gb"] = _ajustar_gb(ds, meta) if meta["usa_gb"] else None
    pesos = json.loads((PESOS / f"pesos_{ds}.json").read_text(encoding="utf-8"))
    return {"pre": pre, "aux": aux, "pesos": pesos, "cfg": M.CONFIGS[ds], "P": P}


def onehot(ds, filas):
    """filas: DataFrame con las columnas crudas de la fuente. Devuelve X one-hot en el orden de entrenamiento."""
    d = cargar(ds)
    pre, cfg, P = d["pre"], d["cfg"], d["P"]
    df = filas.copy()
    df["Etiqueta"] = "X"
    df = P.limpiar_texto(df, excluir=[cfg["id"], "Etiqueta"])
    df = cfg["derivar"](df).reset_index(drop=True)
    Xc = pre.transform_categorico(df)
    Xg = pre.transform_gaussiano(df)[pre.columnas_gaussiano_]
    partes = {c: Xg[c].to_numpy() for c in pre.columnas_gaussiano_}
    for c in pre.columnas_categorico_:
        if c in Xc.columns and c in pre.categorias_:
            for i, k in enumerate(pre.categorias_[c]):
                partes[f"{c}__{k}"] = (Xc[c] == i).astype(int).to_numpy()
    for c in pre.indicador_nulo_:
        if f"{c}_era_nulo" in Xc.columns:
            partes[f"{c}_era_nulo"] = Xc[f"{c}_era_nulo"].to_numpy()
    return pd.DataFrame(partes).reindex(columns=pre.columnas_onehot_).fillna(0.0)


def _features_gb(P_, K, clases, tau, ordinal):
    pred = P_.argmax(1)
    srt = np.sort(P_, axis=1)
    d = {f"GB_P_{c}": P_[:, k] for k, c in enumerate(clases)}
    d["GB_Margen"] = srt[:, -1] - srt[:, -2]
    d["GB_Entropia"] = -(P_ * np.log(np.clip(P_, 1e-12, None))).sum(axis=1) / np.log(K)
    d["GB_Zona_Duda"] = (P_.max(axis=1) < tau[pred]).astype(int)
    if ordinal:
        d["GB_Esperanza"] = P_ @ np.arange(K)
        for k in range(K - 1):
            d[f"GB_P_Mayor_{k}"] = P_[:, k + 1:].sum(axis=1)
    return pd.DataFrame(d)


def entrada_red(ds, filas):
    """Devuelve la matriz (n, columnas de la red) lista para la red."""
    aux = cargar(ds)["aux"]
    X = onehot(ds, filas).to_numpy(float)
    if aux["gb"] is not None:
        K = len(aux["clases"])
        F = _features_gb(aux["gb"].predict_proba(X), K, aux["clases"], aux["tau"], aux["ordinal"])
        X = np.hstack([X, F[aux["cols_gb"]].to_numpy(float)])
    S = (X - aux["media"]) / aux["escala"]
    if aux["nca"] is not None:
        S = np.hstack([S, S @ aux["nca"].T])
    return pd.DataFrame(S, columns=aux["cols_red"])


def probabilidades(ds, filas):
    """Forward pass de la red: ReLU en ocultas, softmax en salida. Devuelve (n, K)."""
    d = cargar(ds)
    p, T = d["pesos"], d["aux"]["temperatura"]
    a = entrada_red(ds, filas).to_numpy(float)
    capas = len(p["coefs"])
    for i in range(capas):
        z = a @ np.array(p["coefs"][i]) + np.array(p["intercepts"][i])
        a = np.maximum(z, 0) if i < capas - 1 else z
    a = a / T  # calibración por temperatura (T = 1 si no hay archivo de calibración)
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predecir(ds, filas):
    pr = probabilidades(ds, filas)
    clases = cargar(ds)["aux"]["clases"]
    return [clases[i] for i in pr.argmax(1)], pr


def zona_duda(ds, filas):
    """True si la probabilidad máxima de la fila está por debajo del umbral de su clase (zona de duda)."""
    aux = cargar(ds)["aux"]
    if aux["gb"] is None:
        return None
    K = len(aux["clases"])
    F = _features_gb(aux["gb"].predict_proba(onehot(ds, filas).to_numpy(float)), K, aux["clases"], aux["tau"], aux["ordinal"])
    return bool(F["GB_Zona_Duda"].iloc[0])
