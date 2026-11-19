"""Frontera de decisión de una red en el plano de sus dos primeras componentes (PCA).

Cada clase tiene un color: las zonas se pintan con el color de la clase que la red asigna, las líneas
oscuras marcan el borde entre zonas y los puntos son una muestra de validación coloreada por su clase real.
El resto de variables se fija en el promedio de train.
"""
import io
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

import inferencia as I

COLORES = ["#2F5D8C", "#B5652B", "#3E7C5A", "#8A4F7D", "#B0443F", "#7A7D2E"]
G = 120


def nombre(clase):
    return str(clase).replace("_", " ").capitalize()


@lru_cache(maxsize=8)
def _base(ds):
    v4 = I.BI / "datasets_multiclase_v4" / ds
    v2 = I.V2_SALIDA / ds
    Xtr = pd.read_csv(v4 / "X_train_nn.csv").to_numpy(float)
    Xva = pd.read_csv(v4 / "X_val_nn.csv").to_numpy(float)
    yva = pd.read_csv(v2 / "y_val.csv")["Y"].to_numpy()
    pesos = I.cargar(ds)["pesos"]
    W, B = pesos["coefs"], pesos["intercepts"]

    def fwd(a):
        a = np.atleast_2d(a)
        for i in range(len(W)):
            z = a @ np.array(W[i]) + np.array(B[i])
            a = np.maximum(z, 0) if i < len(W) - 1 else z
        return a.argmax(1)

    pca = PCA(2, random_state=42).fit(Xtr)
    Z = pca.transform(Xtr)
    idx = np.random.default_rng(7).choice(len(Xva), size=min(220, len(Xva)), replace=False)
    nube = pca.transform(Xva[idx])
    # Ventana que contiene todos los datos (train y la muestra de validación) con un margen, para que
    # ningún punto quede fuera y las fronteras se vean completas dentro del recuadro.
    U = np.vstack([Z, nube])
    lo, hi = U.min(0), U.max(0)
    margen = 0.03 * (hi - lo)
    lo, hi = lo - margen, hi + margen
    g1, g2 = np.linspace(lo[0], hi[0], G), np.linspace(lo[1], hi[1], G)
    pts = np.array([[a, b] for b in g2 for a in g1])
    grid = fwd(pca.inverse_transform(pts)).reshape(G, G)  # fila = componente 2
    return {"pca": pca, "grid": grid, "lo": lo, "hi": hi, "nube": nube, "clases_y": yva[idx],
            "clases": I.cargar(ds)["aux"]["clases"], "varianza": pca.explained_variance_ratio_}


def figura(ds, nn=None, pred=None):
    """PNG de la frontera: zonas por clase, borde entre zonas, puntos de validación y el dato del usuario."""
    b = _base(ds)
    clases = list(b["clases"])
    k = len(clases)
    cols = COLORES[:k]
    lo, hi = b["lo"], b["hi"]
    n = b["grid"].shape[0]
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=120)
    fig.patch.set_facecolor("#F4F6F9")
    ax.set_facecolor("#F4F6F9")
    ax.contourf(X, Y, b["grid"], levels=np.arange(k + 1) - 0.5, colors=cols, alpha=0.30)
    ax.contour(X, Y, b["grid"], levels=[i - 0.5 for i in range(1, k)], colors="#15233A", linewidths=1.6)

    nube, y_real = b["nube"], np.asarray(b["clases_y"]).astype(int)
    for c in range(k):
        m = y_real == c
        ax.scatter(nube[m, 0], nube[m, 1], s=40, c=cols[c], edgecolors="#15233A", linewidths=0.7, zorder=3)

    if nn is not None:
        z = b["pca"].transform(np.asarray(nn, float).reshape(1, -1))[0]
        ex = min(max(z[0], lo[0]), hi[0])
        ey = min(max(z[1], lo[1]), hi[1])
        idx = clases.index(pred) if pred in clases else 0
        ax.scatter([ex], [ey], s=280, c=[cols[idx]], edgecolors="#15233A", linewidths=2.4, zorder=5)
        ax.annotate(f"Su dato: {nombre(pred or '')}", (ex, ey), xytext=(14, 14), textcoords="offset points",
                    fontsize=10, color="#15233A", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F4F6F9", ec="#15233A"))

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    v1, v2 = b["varianza"]
    ax.set_xlabel(f"Componente 1 ({v1 * 100:.1f} % de la varianza)", color="#4A5B70")
    ax.set_ylabel(f"Componente 2 ({v2 * 100:.1f} %)", color="#4A5B70")
    ax.tick_params(colors="#4A5B70", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#8C9BAD")
    manejadores = [Patch(facecolor=cols[i], alpha=0.6, edgecolor="#15233A", label=nombre(c)) for i, c in enumerate(clases)]
    manejadores.append(Line2D([], [], marker="o", ls="", markerfacecolor="#9AA7B6", markeredgecolor="#15233A",
                              label="Validación (clase real)"))
    ax.legend(handles=manejadores, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(manejadores),
              frameon=False, fontsize=9, labelcolor="#15233A")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
