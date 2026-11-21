"""T16: ds5 Keras network (variant B, sqrt-weighted CE) with and without payment-timing proxies.

Proxies are built with the strict rule (payments strictly before Corte, see t15), median-imputed with train
medians, given a missing-value flag, and standardised with train mean and std before being appended to the
112 network inputs. Order: 3 seeds per variant, selection on validation macro F1 only, then test is read once
for the selected variant and for the base (paired bootstrap of the difference). Writes
resultados_keras/t16_ds5_keras_microatraso.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402

BOOT = 1000


def macro(y, pred, K=4):
    return f1_score(y, pred, average="macro", labels=range(K), zero_division=0)


def main():
    X, y = t4.cargar_datos(t4.DS5)
    pesos, cfg = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    K, idx = len(clases), clases.index("MORA_LEVE")

    d, F = t14.construir(estricto=True)
    parte = d.Particion.to_numpy()
    cols = list(F.columns)
    med = F[parte == "train"].median()
    falta = F.isna().to_numpy().astype(np.float32)[:, [cols.index("n_fechas")]]
    Fi = F.fillna(med)
    mu, sd = Fi[parte == "train"].mean(), Fi[parte == "train"].std().replace(0, 1.0)
    Z = ((Fi - mu) / sd).to_numpy(np.float32)
    Z = np.hstack([Z, falta])
    Xm = {p: np.hstack([X[p], Z[parte == p]]) for p in ("train", "val", "test")}
    print("entradas base", X["train"].shape[1], "con proxies", Xm["train"].shape[1], flush=True)

    w = t4.pesos_sqrt_balanceados(y["train"], K)
    modelos = {}
    val = {}
    for nombre, XX in (("base", X), ("proxies", Xm)):
        filas, mods = [], []
        for semilla in t4.SEMILLAS:
            m = t4.entrenar(XX["train"], y["train"], XX["val"], y["val"], K, cfg["alpha"], "wce", semilla,
                            pesos_clase=w)
            mods.append(m)
            filas.append(t4.metricas(y["val"], t4.probas(m, XX["val"])))
        modelos[nombre] = mods
        val[nombre] = {
            "macro_f1": round(float(np.mean([f["macro_f1"] for f in filas])), 4),
            "desv": round(float(np.std([f["macro_f1"] for f in filas])), 4),
            "f1_leve": round(float(np.mean([f["f1_clase"][idx] for f in filas])), 4),
            "f1_clase": [round(float(np.mean([f["f1_clase"][k] for f in filas])), 4) for k in range(K)],
            "corridas": filas}
        print(nombre, "val", {k: v for k, v in val[nombre].items() if k != "corridas"}, flush=True)

    elegido = max(val, key=lambda k: val[k]["macro_f1"])
    print("elegido por validacion:", elegido, flush=True)

    rng = np.random.default_rng(0)
    prueba = {}
    for nombre, XX in (("base", X), ("proxies", Xm)):
        prueba[nombre] = [t4.probas(m, XX["test"]) for m in modelos[nombre]]
    yt = y["test"]
    por_semilla = []
    for i, semilla in enumerate(t4.SEMILLAS):
        p0, p1 = prueba["base"][i].argmax(1), prueba["proxies"][i].argmax(1)
        dm, dl = [], []
        for _ in range(BOOT):
            s = rng.integers(0, len(yt), len(yt))
            dm.append(macro(yt[s], p1[s]) - macro(yt[s], p0[s]))
            dl.append(f1_score(yt[s], p1[s], average=None, labels=range(K), zero_division=0)[idx]
                      - f1_score(yt[s], p0[s], average=None, labels=range(K), zero_division=0)[idx])
        por_semilla.append({
            "semilla": semilla,
            "base": t4.metricas(yt, prueba["base"][i]), "proxies": t4.metricas(yt, prueba["proxies"][i]),
            "boot_macro_ic95": [round(float(np.percentile(dm, 2.5)), 4), round(float(np.percentile(dm, 97.5)), 4)],
            "boot_leve_ic95": [round(float(np.percentile(dl, 2.5)), 4), round(float(np.percentile(dl, 97.5)), 4)]})
        print(json.dumps(por_semilla[-1]), flush=True)

    res = {"clases": clases, "proxies": cols, "entradas": Xm["train"].shape[1], "validacion": val,
           "elegido_por_validacion": elegido, "prueba_por_semilla": por_semilla}
    (t4.SALIDA / "t16_ds5_keras_microatraso.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
