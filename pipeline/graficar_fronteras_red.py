"""Decision regions of the real ANN, drawn on the plane of the first two PCA components of its input.

Unlike nucleo/frontera.py, the regions come from the trained network itself: a numpy forward pass with the
saved weights (resultados_convergencia/pesos_<ds>.json) and the saved temperature. Each point of the plane is
mapped back to the full input space with PCA.inverse_transform, so the remaining input dimensions stay at
their PCA mean. The plane is therefore a slice of the real decision function, not a surrogate model.

Output per dataset: graficos/red_<ds>.png with two panels: class regions and normalized entropy.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

PROGRAMA = Path(__file__).resolve().parents[1]
V4 = PROGRAMA.parent  # datasets_multiclase_v4 holds the dsN folders next to programa_ann
PESOS = PROGRAMA / "resultados_convergencia"
AUX = PROGRAMA / "auxiliares"
SALIDA = PROGRAMA / "graficos"

DATASETS = ["ds5_pagos_multimes", "ds7_historico_garantias"]
GRID = 320
N_ENTRENO = 6000
RECORTE = 0.5
SEMILLA = 42
COLORES = ["#2F5D8C", "#B5652B", "#3E7C5A", "#8A4F7D", "#B0443F", "#7A7D2E"]


def nombre_salida(ds):
    return f"red_{ds}.png"


def cargar_red(ds):
    pesos = json.loads((PESOS / f"pesos_{ds}.json").read_text(encoding="utf-8"))
    aux = json.loads((AUX / f"{ds}.json").read_text(encoding="utf-8"))
    ruta_cal = AUX / f"calibracion_{ds}.json"
    temperatura = json.loads(ruta_cal.read_text(encoding="utf-8"))["temperatura"] if ruta_cal.exists() else 1.0
    return pesos, aux, temperatura


def propagar(X, pesos, temperatura):
    """Forward pass: ReLU in hidden layers, softmax at the output, divided by the calibration temperature."""
    a = X
    capas = len(pesos["coefs"])
    for i in range(capas):
        z = a @ np.asarray(pesos["coefs"][i]) + np.asarray(pesos["intercepts"][i])
        a = np.maximum(z, 0) if i < capas - 1 else z
    a = a / temperatura
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def entropia_normalizada(P):
    K = P.shape[1]
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(axis=1) / np.log(K)


def dibujar(ds):
    pesos, aux, temperatura = cargar_red(ds)
    clases = list(aux["clases"])
    K = len(clases)

    Xtr = pd.read_csv(V4 / ds / "X_train_nn.csv").to_numpy(float)
    Xva = pd.read_csv(V4 / ds / "X_val_nn.csv").to_numpy(float)
    yva = pd.read_csv(V4 / ds / "y_val.csv")["Y"].to_numpy().astype(int)

    entradas = len(pesos["coefs"][0])
    if Xva.shape[1] != entradas:
        raise ValueError(f"{ds}: the net expects {entradas} inputs, the CSV has {Xva.shape[1]}")
    if yva.max() >= K:
        raise ValueError(f"{ds}: label {yva.max()} outside the {K} classes")

    rng = np.random.default_rng(SEMILLA)
    idx = rng.choice(len(Xtr), size=min(N_ENTRENO, len(Xtr)), replace=False)
    pca = PCA(2, random_state=SEMILLA).fit(Xtr[idx])
    Zva = pca.transform(Xva)
    lo = np.percentile(Zva, RECORTE, axis=0)
    hi = np.percentile(Zva, 100 - RECORTE, axis=0)
    margen = 0.05 * (hi - lo)
    lo, hi = lo - margen, hi + margen
    fuera = float(np.mean(np.any((Zva < lo) | (Zva > hi), axis=1)))

    xs = np.linspace(lo[0], hi[0], GRID)
    ys = np.linspace(lo[1], hi[1], GRID)
    gx, gy = np.meshgrid(xs, ys)
    plano = np.column_stack([gx.ravel(), gy.ravel()])
    Pg = propagar(pca.inverse_transform(plano), pesos, temperatura)
    zona = Pg.argmax(1).reshape(GRID, GRID)
    Hn = entropia_normalizada(Pg).reshape(GRID, GRID)

    Pv = propagar(Xva, pesos, temperatura)
    pred = Pv.argmax(1)
    bal = balanced_accuracy_score(yva, pred)
    macro_f1 = f1_score(yva, pred, average="macro")

    cols = COLORES[:K]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=120)
    fig.patch.set_facecolor("#F4F6F9")

    ax1.set_facecolor("#F4F6F9")
    ax1.contourf(gx, gy, zona, levels=np.arange(K + 1) - 0.5, colors=cols, alpha=0.30)
    ax1.contour(gx, gy, zona, levels=[i - 0.5 for i in range(1, K)], colors="#15233A", linewidths=1.6)
    dentro = np.all((Zva >= lo) & (Zva <= hi), axis=1)
    acierto = pred == yva
    for c in range(K):
        m = (yva == c) & dentro & acierto
        ax1.scatter(Zva[m, 0], Zva[m, 1], s=22, c=cols[c], edgecolors="#15233A", linewidths=0.5, zorder=3)
        m = (yva == c) & dentro & ~acierto
        ax1.scatter(Zva[m, 0], Zva[m, 1], s=34, c=cols[c], edgecolors="#B0443F", linewidths=1.6, zorder=4)
    ax1.set_title("Regiones de decisión de la red", color="#15233A", loc="left")

    ax2.set_facecolor("#F4F6F9")
    mapa = ax2.contourf(gx, gy, Hn, levels=np.linspace(0, 1, 21), cmap="magma_r")
    fig.colorbar(mapa, ax=ax2, label="Entropía normalizada H / ln K")
    ax2.set_title("Incertidumbre de la red (entropía normalizada)", color="#15233A", loc="left")

    for ax in (ax1, ax2):
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        v1, v2 = pca.explained_variance_ratio_
        ax.set_xlabel(f"Componente PCA 1 ({v1 * 100:.1f} % de la varianza de entrada)", color="#4A5B70")
        ax.set_ylabel(f"Componente PCA 2 ({v2 * 100:.1f} % de la varianza de entrada)", color="#4A5B70")
        ax.tick_params(colors="#4A5B70", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#8C9BAD")

    leyenda = [Patch(facecolor=cols[i], alpha=0.6, edgecolor="#15233A", label=c.replace("_", " ").capitalize())
               for i, c in enumerate(clases)]
    leyenda.append(Line2D([], [], marker="o", ls="", markerfacecolor="#9AA7B6", markeredgecolor="#15233A",
                          label="Validación, acierto"))
    leyenda.append(Line2D([], [], marker="o", ls="", markerfacecolor="#9AA7B6", markeredgecolor="#B0443F",
                          markeredgewidth=1.6, label="Validación, error"))
    ax1.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(1.0, -0.14), ncol=len(leyenda),
               frameon=False, fontsize=8.5)

    fig.suptitle(f"{ds}: exactitud equilibrada {bal * 100:.1f} %, macro F1 {macro_f1 * 100:.1f} % en validación",
                 color="#15233A", fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "Corte real de la red: el plano PCA de sus entradas, con el resto de dimensiones en la media. "
             f"Fuera de la ventana: {fuera * 100:.0f} % de la validación. Entrenada solo con train.",
             fontsize=8.5, color="#4A5B70")

    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA / nombre_salida(ds)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return {"ds": ds, "ruta": str(ruta), "bal_val": round(bal, 4), "macro_f1_val": round(macro_f1, 4),
            "fuera_ventana": round(fuera, 4)}


def main():
    for ds in DATASETS:
        print(dibujar(ds))


if __name__ == "__main__":
    sys.exit(main())
