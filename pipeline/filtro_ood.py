"""Geometric out-of-distribution filters for the ds7 ensemble. NumPy and scikit-learn only.

Step 1 of the cascade: a case farther from the training support than d_max is rejected as out of range
before the softmax is read. d_max is the 99th percentile of the score on validation, so about 1 % of
legitimate cases is rejected by design.

Limitation: these filters see anomalies in input space. They do not see moderate drift inside the support
or a change in the class priors P(y).
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

PERCENTIL = 99.0


class FiltroKNN:
    """Score = mean Euclidean distance to the k nearest training rows. No distributional assumption,
    so it works on binary and one-hot columns."""

    def __init__(self, k=5):
        self.k = k

    def ajustar(self, X):
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def puntaje(self, X):
        d, _ = self.nn_.kneighbors(X)
        return d.mean(1)


class FiltroMahalanobis:
    """Class-conditional Mahalanobis (shared covariance) with Ledoit-Wolf shrinkage, so a singular sample
    covariance is still invertible. Score = smallest squared distance over classes."""

    def ajustar(self, Z, y, K):
        self.medias_ = np.stack([Z[y == k].mean(0) for k in range(K)])
        centrado = Z - self.medias_[y]
        self.precision_ = LedoitWolf().fit(centrado).precision_
        return self

    def puntaje(self, Z):
        d2 = []
        for mu in self.medias_:
            c = Z - mu
            d2.append(np.einsum("ij,jk,ik->i", c, self.precision_, c))
        return np.min(d2, axis=0)


def penultima(modelos, X):
    """Concatenated hidden-layer activations (Dense + ReLU) of every member, computed from its weights."""
    partes = []
    for m in modelos:
        W, b = m.layers[0].get_weights()
        partes.append(np.maximum(X @ W + b, 0.0))
    return np.hstack(partes)


def umbral_distancia(puntajes_val, percentil=PERCENTIL):
    return float(np.percentile(puntajes_val, percentil))


def rechaza(puntajes, d_max):
    return np.asarray(puntajes) > d_max
