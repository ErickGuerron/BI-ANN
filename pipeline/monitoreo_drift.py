"""Population-level drift monitor for a window of recent cases. NumPy only.

A per-case filter cannot see moderate drift inside the training support or a change in the class priors.
This monitor compares a WINDOW of cases with a reference and raises an alert on four statistics:

  psi_max   maximum PSI over the continuous input columns (deciles of the training data)
  psi_pred  PSI of the predicted-class distribution (prior shift)
  tasa_ood  share of cases rejected by the geometric OOD filter
  desv_amb  absolute deviation of the ambiguity share from its reference

With a guarantee reference (g_train, e_train) it adds two more, with analytic thresholds:
  tasa_garantia_desv  z-score of the share of rows that carry a guarantee
  ks_cobertura        scaled KS on log(Valor_Garantiza / Valor_Entregado) among those rows

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


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic: largest gap between the two empirical CDFs."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    todo = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, todo, side="right") / len(a)
                        - np.searchsorted(b, todo, side="right") / len(b)).max())


CLAVES = ("psi_max", "psi_pred", "tasa_ood", "desv_amb")
CLAVES_GARANTIA = ("tasa_garantia_desv", "ks_cobertura")
MIN_POSITIVAS = 5
# Analytic thresholds for the guarantee statistics (two-sided alpha about 0.005 each). They do not come from
# resampled validation windows: only ~8 % of the rows have a guarantee, so a pool of a few hundred positives
# would bake its own offset from the training data into the threshold.
Z_UMBRAL = 2.8     # |share - reference share| / sqrt(p0 (1 - p0) / n)
KS_UMBRAL = 1.73   # KS * sqrt(n1 n2 / (n1 + n2)); Smirnov limit for alpha = 0.005
UMBRALES_GARANTIA = {"tasa_garantia_desv": Z_UMBRAL, "ks_cobertura": KS_UMBRAL}


def log_cobertura(g, e):
    """log(Valor_Garantiza / Valor_Entregado) over the rows that have a guarantee (g > 0 and e > 0)."""
    g, e = np.asarray(g, float), np.asarray(e, float)
    m = (g > 0) & (e > 0)
    return np.log(g[m] / e[m])


class Referencia:
    """Reference built once from training inputs and validation outputs.

    With g_train and e_train (raw Valor_Garantiza and Valor_Entregado) it also watches the guarantee, the input the
    ds7 network leans on most: the share of rows that carry one, and the distribution of their coverage
    (KS on log(g / e)). Only about 8 % of the rows have a guarantee, so a PSI over the whole column is nearly
    blind; these two statistics look at the share and at the positive amounts separately."""

    def __init__(self, Xtr, cont, pred_val, ood_val, amb_val, K, g_train=None, e_train=None):
        self.cont = list(cont)
        self.K = K
        self.bordes_ = [bordes(Xtr[:, j]) for j in self.cont]
        self.props_ = [proporciones(Xtr[:, j], b) for b, j in zip(self.bordes_, self.cont)]
        self.pred_ = np.bincount(pred_val, minlength=K) / len(pred_val)
        self.tasa_ood_ = float(np.mean(ood_val))
        self.tasa_amb_ = float(np.mean(amb_val))
        self.garantia = g_train is not None
        if self.garantia:
            self.tasa_garantia_ = float(np.mean(np.asarray(g_train) > 0))
            self.logk_ref_ = np.sort(log_cobertura(g_train, e_train))

    @property
    def claves(self):
        return CLAVES + (CLAVES_GARANTIA if self.garantia else ())

    def estadisticos(self, X, pred, ood, amb, g=None, e=None):
        psis = [psi(p, proporciones(X[:, j], b)) for p, b, j in zip(self.props_, self.bordes_, self.cont)]
        est = {
            "psi_max": float(max(psis)),
            "columna_psi_max": int(self.cont[int(np.argmax(psis))]),
            "psi_pred": psi(self.pred_, np.bincount(pred, minlength=self.K) / len(pred)),
            "tasa_ood": float(np.mean(ood)),
            "desv_amb": float(abs(np.mean(amb) - self.tasa_amb_)),
        }
        if self.garantia:
            p0, n = self.tasa_garantia_, len(g)
            est["tasa_garantia_desv"] = float(abs(np.mean(np.asarray(g) > 0) - p0) / np.sqrt(p0 * (1 - p0) / n))
            logk = log_cobertura(g, e)
            n1, n2 = len(logk), len(self.logk_ref_)
            est["ks_cobertura"] = (ks(logk, self.logk_ref_) * float(np.sqrt(n1 * n2 / (n1 + n2)))
                                   if n1 >= MIN_POSITIVAS else 0.0)
        return est


def ventana(rng, n, *arrays):
    """Window of n cases resampled with replacement; every array is indexed with the same draw."""
    idx = rng.choice(len(arrays[0]), size=n, replace=True)
    return tuple(a[idx] for a in arrays)


def calibrar_umbrales(ref, X, pred, ood, amb, n, n_ventanas, cuantil, rng, g=None, e=None):
    """Empirical null thresholds for window size n from windows resampled out of a clean dataset."""
    arrays = (X, pred, ood, amb) + ((g, e) if ref.garantia else ())
    est = [ref.estadisticos(*ventana(rng, n, *arrays)) for _ in range(n_ventanas)]
    umbrales = {k: float(np.quantile([x[k] for x in est], cuantil)) for k in CLAVES}
    if ref.garantia:
        umbrales.update(UMBRALES_GARANTIA)
    return umbrales


def alerta(est, umbrales):
    disparadas = [k for k in umbrales if est[k] > umbrales[k]]
    return {"alerta": bool(disparadas), "disparadas": disparadas}
