"""Frontera de decisión en el plano de las dos primeras componentes principales (PCA).

Los puntos son casos reales de validación proyectados al plano PCA, que conserva las distancias.
La frontera la dibuja un MLP de dos entradas entrenado solo con esas componentes y solo con train.
No es la red de la app (que usa todas las variables): es una aproximación en 2D que muestra la forma
de las zonas de decisión. La ventana se recorta en percentiles para que los atípicos no aplasten el gráfico.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import inferencia as I

COLORES = ["#2F5D8C", "#B5652B", "#3E7C5A", "#8A4F7D", "#B0443F", "#7A7D2E"]
G = 240              # resolución del fondo
N_ENTRENO = 4000     # casos de train para ajustar PCA y el modelo de 2 entradas
N_VALIDACION = 400   # casos de validación dibujados
RECORTE = 0.5        # percentil para la ventana (0.5 y 99.5)
SEMILLA = 42


def nombre(clase):
    return str(clase).replace("_", " ").capitalize()


@lru_cache(maxsize=8)
def _base(ds):
    v4 = I.BI / "datasets_multiclase_v4" / ds
    v2 = I.V2_SALIDA / ds
    Xtr = pd.read_csv(v4 / "X_train_nn.csv").to_numpy(float)
    ytr = pd.read_csv(v2 / "y_train.csv")["Y"].to_numpy().astype(int)
    Xva = pd.read_csv(v4 / "X_val_nn.csv").to_numpy(float)
    yva = pd.read_csv(v2 / "y_val.csv")["Y"].to_numpy().astype(int)

    rng = np.random.default_rng(SEMILLA)
    i_tr = rng.choice(len(Xtr), size=min(N_ENTRENO, len(Xtr)), replace=False)
    Xs, ys = Xtr[i_tr], ytr[i_tr]

    pca = PCA(2, random_state=SEMILLA).fit(Xs)
    Zs = pca.transform(Xs)
    Zva = pca.transform(Xva)

    modelo = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16, 16), alpha=1e-3, max_iter=2000, early_stopping=True,
                      random_state=SEMILLA),
    ).fit(Zs, ys)
    exactitud_val = balanced_accuracy_score(yva, modelo.predict(Zva))

    lo = np.percentile(Zs, RECORTE, axis=0)
    hi = np.percentile(Zs, 100 - RECORTE, axis=0)
    margen = 0.05 * (hi - lo)
    lo, hi = lo - margen, hi + margen
    fuera = np.mean(np.any((Zva < lo) | (Zva > hi), axis=1))

    i_va = rng.choice(len(Xva), size=min(N_VALIDACION, len(Xva)), replace=False)
    return {"pca": pca, "modelo": modelo, "lo": lo, "hi": hi, "Zva": Zva[i_va], "yva": yva[i_va],
            "exactitud_val": exactitud_val, "fuera": fuera, "clases": I.cargar(ds)["aux"]["clases"]}


def figura(ds, nn=None, pred=None):
    """PNG de la frontera PCA: zonas del MLP de 2 entradas, casos reales de validación y el dato nuevo."""
    b = _base(ds)
    clases = list(b["clases"])
    k = len(clases)
    cols = COLORES[:k]
    lo, hi = b["lo"], b["hi"]

    xs = np.linspace(lo[0], hi[0], G)
    ys_grid = np.linspace(lo[1], hi[1], G)
    X, Y = np.meshgrid(xs, ys_grid)
    zona = b["modelo"].predict(np.column_stack([X.ravel(), Y.ravel()])).reshape(G, G)

    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=120)
    fig.patch.set_facecolor("#F4F6F9")
    ax.set_facecolor("#F4F6F9")
    ax.contourf(X, Y, zona, levels=np.arange(k + 1) - 0.5, colors=cols, alpha=0.30)
    ax.contour(X, Y, zona, levels=[i - 0.5 for i in range(1, k)], colors="#15233A", linewidths=1.8)

    Z, yv = b["Zva"], np.asarray(b["yva"])
    dentro = np.all((Z >= lo) & (Z <= hi), axis=1)
    for c in range(k):
        m = (yv == c) & dentro
        ax.scatter(Z[m, 0], Z[m, 1], s=40, c=cols[c], edgecolors="#15233A", linewidths=0.7, zorder=3)

    if nn is not None:
        z = b["pca"].transform(np.asarray(nn, float).reshape(1, -1))[0]
        ex = min(max(z[0], lo[0]), hi[0])
        ey = min(max(z[1], lo[1]), hi[1])
        i_clase = clases.index(pred) if pred in clases else 0
        ax.scatter([ex], [ey], s=300, marker="*", c=[cols[i_clase]], edgecolors="#15233A", linewidths=1.8, zorder=5)
        ax.annotate(f"Su dato: {nombre(pred or '')}", (ex, ey), xytext=(14, 14), textcoords="offset points",
                    fontsize=10, color="#15233A", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F4F6F9", ec="#15233A"))

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    v1, v2 = b["pca"].explained_variance_ratio_
    ax.set_xlabel(f"Componente 1 ({v1 * 100:.1f} % de la varianza)", color="#4A5B70")
    ax.set_ylabel(f"Componente 2 ({v2 * 100:.1f} % de la varianza)", color="#4A5B70")
    ax.tick_params(colors="#4A5B70", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#8C9BAD")

    manejadores = [Patch(facecolor=cols[i], alpha=0.6, edgecolor="#15233A", label=nombre(c))
                   for i, c in enumerate(clases)]
    manejadores.append(Line2D([], [], marker="o", ls="", markerfacecolor="#9AA7B6", markeredgecolor="#15233A",
                              label="Validación (clase real)"))
    ax.legend(handles=manejadores, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(manejadores),
              frameon=False, fontsize=9, labelcolor="#15233A")
    fig.text(0.5, -0.06,
             f"Fondo: red de 2 entradas (componentes PCA) entrenada solo con train; exactitud equilibrada en validación "
             f"{b['exactitud_val'] * 100:.1f} %. No es la red de la app. Fuera de la ventana: {b['fuera'] * 100:.0f} % "
             f"de la validación.",
             ha="center", fontsize=8.5, color="#4A5B70")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
