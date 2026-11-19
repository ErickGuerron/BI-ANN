"""Mapa de clases de un dataset en un plano euclidiano 2D (t-SNE de datos reales).

Los puntos son casos reales (entrenamiento y validación), proyectados con PCA previa y t-SNE, como en el
control de MNIST. Cada clase tiene su color.

El fondo lo pinta un clasificador AUXILIAR (SVM) entrenado solo sobre las coordenadas t-SNE de train.
No es la red de la app: t-SNE no genera una función de decisión. Por eso se rotula como aproximación.
El dato nuevo se ubica de forma aproximada con sus vecinos más cercanos en el espacio original, porque
t-SNE no proyecta puntos nuevos.
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
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

import inferencia as I

COLORES = ["#2F5D8C", "#B5652B", "#3E7C5A", "#8A4F7D", "#B0443F", "#7A7D2E"]
G = 220              # resolución del fondo
N_ENTRENO = 2500     # casos de train usados para t-SNE y para el clasificador auxiliar
N_VALIDACION = 300   # casos de validación dibujados
K_VECINOS = 10       # vecinos para ubicar el dato nuevo
SEMILLA = 42
CARPETA_CACHE = I.PROGRAMA / "graficos" / "cache"


def nombre(clase):
    return str(clase).replace("_", " ").capitalize()


def _tsne_con_cache(ds, Z):
    """t-SNE a 2D sobre Z (PCA previa). Se guarda en caché: el cálculo tarda y es determinista."""
    CARPETA_CACHE.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_CACHE / f"tsne2d_{ds}_{Z.shape[0]}.npy"
    if ruta.exists():
        return np.load(ruta)
    perplejidad = min(30, (Z.shape[0] - 1) / 3)
    E = TSNE(n_components=2, perplexity=perplejidad, init="pca", learning_rate="auto",
             random_state=SEMILLA).fit_transform(Z)
    np.save(ruta, E)
    return E


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
    i_va = rng.choice(len(Xva), size=min(N_VALIDACION, len(Xva)), replace=False)
    Xs, ys = Xtr[i_tr], ytr[i_tr]
    Xv, yv = Xva[i_va], yva[i_va]

    pca = PCA(min(50, Xs.shape[1]), random_state=SEMILLA).fit(Xs)
    Zs, Zv = pca.transform(Xs), pca.transform(Xv)
    E = _tsne_con_cache(ds, np.vstack([Zs, Zv]))
    Es, Ev = E[: len(Zs)], E[len(Zs):]

    aux = SVC(kernel="rbf", C=10, gamma="scale").fit(Es, ys)
    exactitud_val = balanced_accuracy_score(yv, aux.predict(Ev))

    vecinos = NearestNeighbors(n_neighbors=K_VECINOS).fit(Zs)

    U = E
    lo, hi = U.min(0), U.max(0)
    margen = 0.04 * (hi - lo)
    return {"pca": pca, "aux": aux, "vecinos": vecinos, "Es": Es, "ys": ys, "Ev": Ev, "yv": yv,
            "lo": lo - margen, "hi": hi + margen, "exactitud_val": exactitud_val,
            "clases": I.cargar(ds)["aux"]["clases"]}


def figura(ds, nn=None, pred=None):
    """PNG del mapa de clases: fondo del clasificador auxiliar, casos reales y el dato nuevo aproximado."""
    b = _base(ds)
    clases = list(b["clases"])
    k = len(clases)
    cols = COLORES[:k]
    lo, hi = b["lo"], b["hi"]

    xs = np.linspace(lo[0], hi[0], G)
    ys_grid = np.linspace(lo[1], hi[1], G)
    X, Y = np.meshgrid(xs, ys_grid)
    zona = b["aux"].predict(np.column_stack([X.ravel(), Y.ravel()])).reshape(G, G)

    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=120)
    fig.patch.set_facecolor("#F4F6F9")
    ax.set_facecolor("#F4F6F9")
    ax.contourf(X, Y, zona, levels=np.arange(k + 1) - 0.5, colors=cols, alpha=0.28)
    ax.contour(X, Y, zona, levels=[i - 0.5 for i in range(1, k)], colors="#15233A", linewidths=1.4)

    Ev, yv = b["Ev"], np.asarray(b["yv"])
    for c in range(k):
        m = yv == c
        ax.scatter(Ev[m, 0], Ev[m, 1], s=34, c=cols[c], edgecolors="#15233A", linewidths=0.6, zorder=3)

    if nn is not None:
        z = b["pca"].transform(np.asarray(nn, float).reshape(1, -1))
        _, idx = b["vecinos"].kneighbors(z)
        ex, ey = b["Es"][idx[0]].mean(0)
        i_clase = clases.index(pred) if pred in clases else 0
        ax.scatter([ex], [ey], s=320, marker="*", c=[cols[i_clase]], edgecolors="#15233A", linewidths=1.6, zorder=5)
        ax.annotate(f"Su dato (aprox.): {nombre(pred or '')}", (ex, ey), xytext=(14, 14),
                    textcoords="offset points", fontsize=10, color="#15233A", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F4F6F9", ec="#15233A"))

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("Dimensión t-SNE 1 (sin unidades)", color="#4A5B70")
    ax.set_ylabel("Dimensión t-SNE 2 (sin unidades)", color="#4A5B70")
    ax.tick_params(colors="#4A5B70", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#8C9BAD")

    manejadores = [Patch(facecolor=cols[i], alpha=0.6, edgecolor="#15233A", label=nombre(c))
                   for i, c in enumerate(clases)]
    manejadores.append(Line2D([], [], marker="o", ls="", markerfacecolor="#9AA7B6", markeredgecolor="#15233A",
                              label="Validación (clase real)"))
    ax.legend(handles=manejadores, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(manejadores),
              frameon=False, fontsize=9, labelcolor="#15233A")
    fig.text(0.5, -0.04, f"Fondo: clasificador auxiliar (SVM) sobre t-SNE, exactitud equilibrada en validación "
             f"{b['exactitud_val'] * 100:.1f} %. No es la red de la app.",
             ha="center", fontsize=8.5, color="#4A5B70")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
