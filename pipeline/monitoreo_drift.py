"""Population-level drift monitor for a window of recent cases. NumPy only.

A per-case filter cannot see moderate drift inside the training support or a change in the class priors.
This monitor compares a WINDOW of cases with a reference and raises an alert on four statistics:

  psi_max   maximum PSI over the continuous input columns (deciles of the training data)
  psi_pred  PSI of the predicted-class distribution (prior shift)
  tasa_ood  share of cases rejected by the geometric OOD filter
  desv_amb  absolute deviation of the ambiguity share from its reference

PSI noise grows as the window shrinks (about (bins - 1) / n for a clean window), so fixed 0.1 / 0.25 limits
would fire all the time on small windows. Each threshold is therefore calibrated per window size from
resampled validation windows (empirical null) at a chosen quantile.
"""
import numpy as np

EPS = 1e-4
BINS = 10


def bordes(x, bins=BINS):
    """Interior decile edges of a reference column, taken from observed values (no interpolation).
    Bins are (-inf, e1], (e1, e2], ...; an edge at the column maximum would leave an empty top bin and is
    dropped, so a binary column gets exactly one edge."""
    q = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1], method="lower"))
    return q[q < np.max(x)]


def proporciones(x, b):
    idx = np.searchsorted(b, x, side="left")
    cuenta = np.bincount(idx, minlength=len(b) + 1).astype(float)
    return cuenta / cuenta.sum()


def psi(ref, act):
    ref = np.clip(ref, EPS, None)
    act = np.clip(act, EPS, None)
    return float(((act - ref) * np.log(act / ref)).sum())


class Referencia:
    """Reference built once from training inputs and validation outputs."""

    def __init__(self, Xtr, cont, pred_val, ood_val, amb_val, K):
        self.cont = list(cont)
        self.K = K
        self.bordes_ = [bordes(Xtr[:, j]) for j in self.cont]
        self.props_ = [proporciones(Xtr[:, j], b) for b, j in zip(self.bordes_, self.cont)]
        self.pred_ = np.bincount(pred_val, minlength=K) / len(pred_val)
        self.tasa_ood_ = float(np.mean(ood_val))
        self.tasa_amb_ = float(np.mean(amb_val))

    def estadisticos(self, X, pred, ood, amb):
        psis = [psi(p, proporciones(X[:, j], b)) for p, b, j in zip(self.props_, self.bordes_, self.cont)]
        return {
            "psi_max": float(max(psis)),
            "columna_psi_max": int(self.cont[int(np.argmax(psis))]),
            "psi_pred": psi(self.pred_, np.bincount(pred, minlength=self.K) / len(pred)),
            "tasa_ood": float(np.mean(ood)),
            "desv_amb": float(abs(np.mean(amb) - self.tasa_amb_)),
        }


CLAVES = ("psi_max", "psi_pred", "tasa_ood", "desv_amb")


def ventana(rng, n, X, pred, ood, amb):
    idx = rng.choice(len(X), size=n, replace=True)
    return X[idx], pred[idx], ood[idx], amb[idx]


def calibrar_umbrales(ref, X, pred, ood, amb, n, n_ventanas, cuantil, rng):
    """Empirical null thresholds for window size n from windows resampled out of a clean dataset."""
    est = [ref.estadisticos(*ventana(rng, n, X, pred, ood, amb)) for _ in range(n_ventanas)]
    return {k: float(np.quantile([e[k] for e in est], cuantil)) for k in CLAVES}


def alerta(est, umbrales):
    disparadas = [k for k in CLAVES if est[k] > umbrales[k]]
    return {"alerta": bool(disparadas), "disparadas": disparadas}
