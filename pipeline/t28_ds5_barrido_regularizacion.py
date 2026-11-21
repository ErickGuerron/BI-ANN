"""T28 (ds5, step 4): small regularisation sweep for the network with par + cuotas_atrasadas_max.

Grid: hidden units {16, 32, 64} x L2 alpha {0.3, 1, 3, 10} (12 configs; the app uses 32 and 3.0). Same data,
folds, GB_* refit per fold and class weights as t27, 3 seeds (42-44). Selection uses validation macro F1 only,
averaged over fold x seed pairs, with the one-standard-error rule: among configs within 1 SE of the best, take
the simplest (fewest units, then largest alpha). The test set is read once, only for the chosen config and for
the default (32, 3.0). Writes resultados_keras/t28_barrido_regularizacion.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402
import t18_ds5_gb_por_pliegue as t18  # noqa: E402
import t25_ds5_cuotas_atrasadas as t25  # noqa: E402
from t17_ds5_pliegues_temporales import PLIEGUES  # noqa: E402

SEMILLAS = [42, 43, 44]
CAPAS = [16, 32, 64]
ALFAS = [0.3, 1.0, 3.0, 10.0]
DEFECTO = (32, 3.0)
EXTRA = ["dia_desv", "dia_ult_menos_med", "cuotas_atrasadas_max"]
UNIDADES = {"n": 32}


def construir_con_capa(d_entrada, K, alpha, n_entreno):
    reg = regularizers.L2(0.5 * alpha / n_entreno)
    return keras.Sequential([
        keras.Input(shape=(d_entrada,)),
        layers.Dense(UNIDADES["n"], activation="relu", kernel_regularizer=reg),
        layers.Dense(K, activation="softmax", kernel_regularizer=reg),
    ])


t4.construir = construir_con_capa  # entrenar() resolves construir at call time


def macro(y, P):
    return float(f1_score(y, P.argmax(1), average="macro", labels=range(4), zero_division=0))


def main():
    r3 = json.loads((t18.V3 / "resumen_v3.json").read_text(encoding="utf-8"))
    params, n_iter = r3["hiperparametros_elegidos"], int(r3["iteraciones"])
    X, y = t4.cargar_datos(t4.DS5)
    pesos, _ = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    K, idx = len(clases), clases.index("MORA_LEVE")
    d, F = t14.construir(estricto=True)
    F = pd.concat([F, t25.senales(d)], axis=1)
    orden = ["train", "val", "test"]
    pos = np.concatenate([np.where(d.Particion.to_numpy() == p)[0] for p in orden])
    d, F = d.iloc[pos].reset_index(drop=True), F.iloc[pos].reset_index(drop=True)
    yA = np.concatenate([y[p] for p in orden])
    Xo = np.vstack([pd.read_csv(t18.V2 / f"X_{p}_onehot.csv", float_precision="round_trip").fillna(0).to_numpy(float)
                    for p in orden])
    corte = d.Corte.to_numpy()
    cols = list(F.columns)
    n_oh = Xo.shape[1]

    val = {}   # (capa, alpha) -> list of validation macro F1 per fold x seed
    modelos_test = {}  # (capa, alpha, pliegue) -> test probabilities by seed, kept only to evaluate chosen + default
    for nombre, cf in PLIEGUES.items():
        m = {k: np.isin(corte, v) for k, v in cf.items()}
        gb = t18.gb_pliegue(Xo, yA, corte, m, params, n_iter, K)
        sc = StandardScaler().fit(np.hstack([Xo[m["train"]], gb["train"]]))
        XR = np.zeros((len(Xo), n_oh + len(t18.GB_COLS)), dtype=np.float32)
        for p in orden:
            XR[m[p]] = sc.transform(np.hstack([Xo[m[p]], gb[p]]))
        Fi = F.fillna(F[m["train"]].median())
        mu, sd = Fi[m["train"]].mean(), Fi[m["train"]].std().replace(0, 1.0)
        Zs = ((Fi - mu) / sd).to_numpy(np.float32)
        falta = F.isna().to_numpy().astype(np.float32)[:, [cols.index("n_fechas")]]
        M = np.hstack([XR, Zs[:, [cols.index(c) for c in EXTRA]], falta])
        w = t4.pesos_sqrt_balanceados(yA[m["train"]], K)
        for capa in CAPAS:
            UNIDADES["n"] = capa
            for alpha in ALFAS:
                for semilla in SEMILLAS:
                    mod = t4.entrenar(M[m["train"]], yA[m["train"]], M[m["val"]], yA[m["val"]], K, alpha, "wce",
                                      semilla, pesos_clase=w)
                    val.setdefault((capa, alpha), []).append(macro(yA[m["val"]], t4.probas(mod, M[m["val"]])))
                    if (capa, alpha) == DEFECTO:
                        modelos_test.setdefault((capa, alpha), {})[(nombre, semilla)] = t4.probas(mod, M[m["test"]])
                    else:
                        modelos_test.setdefault((capa, alpha), {})[(nombre, semilla)] = mod.get_weights()
                    keras.backend.clear_session()
                print(nombre, capa, alpha, round(float(np.mean(val[(capa, alpha)][-len(SEMILLAS):])), 4), flush=True)
        # keep fold data needed to score the test set once, later
        modelos_test[("datos", nombre)] = (M[m["test"]], yA[m["test"]])

    tabla = {}
    for (capa, alpha), v in val.items():
        tabla[(capa, alpha)] = (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
    mejor = max(tabla, key=lambda k: tabla[k][0])
    umbral = tabla[mejor][0] - tabla[mejor][1]
    elegibles = [k for k, (mu_, _) in tabla.items() if mu_ >= umbral]
    elegido = sorted(elegibles, key=lambda k: (k[0], -k[1]))[0]
    print("mejor", mejor, "umbral", round(umbral, 4), "elegido", elegido, flush=True)

    # test read once: chosen config (weights re-applied) and default (stored probabilities)
    def prueba(clave):
        filas = []
        for nombre in PLIEGUES:
            Xt, yt = modelos_test[("datos", nombre)]
            for semilla in SEMILLAS:
                guardado = modelos_test[clave][(nombre, semilla)]
                if isinstance(guardado, np.ndarray):
                    P = guardado
                else:
                    UNIDADES["n"] = clave[0]
                    net = construir_con_capa(Xt.shape[1], K, clave[1], 1)
                    net.set_weights(guardado)
                    P = t4.probas(net, Xt)
                f = f1_score(yt, P.argmax(1), average=None, labels=range(4), zero_division=0)
                filas.append({"pliegue": nombre, "semilla": semilla, "macro_f1": float(f.mean()),
                              "f1_leve": float(f[idx]), "f1_clase": f.round(4).tolist()})
        return filas

    salida = {"validacion": {f"{c}|{a}": {"macro_val": round(mu_, 4), "se": round(se, 4)}
                             for (c, a), (mu_, se) in tabla.items()},
              "mejor": list(mejor), "umbral_1se": round(umbral, 4), "elegido": list(elegido), "prueba": {}}
    for nom, clave in (("elegido", elegido), ("defecto", DEFECTO)):
        filas = prueba(clave)
        salida["prueba"][nom] = {"config": list(clave),
                                 "macro_f1": round(float(np.mean([r["macro_f1"] for r in filas])), 4),
                                 "f1_leve": round(float(np.mean([r["f1_leve"] for r in filas])), 4),
                                 "corridas": filas}
        print(nom, clave, salida["prueba"][nom]["macro_f1"], salida["prueba"][nom]["f1_leve"], flush=True)
    (t4.SALIDA / "t28_barrido_regularizacion.json").write_text(json.dumps(salida, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
