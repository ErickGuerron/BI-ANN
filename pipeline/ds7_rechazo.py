"""ds7 (ML puro): detección de soporte + rechazo con umbral calibrado en validación.

1. Soporte. Distancia media a los k vecinos más cercanos del conjunto de entrenamiento (espacio estandarizado).
   Umbral d* = percentil 99 de las distancias de validación. Si una entrada supera d*, sale "fuera de soporte".
2. Calibración. Temperatura T ajustada por NLL en validación sobre log(probabilidades).
3. Rechazo. Se acepta la clase predicha solo si está dentro de soporte y su probabilidad calibrada es >= tau.
   tau es el menor valor de la rejilla que logra precisión >= 95 % sobre los casos aceptados en validación.
4. Prueba se evalúa una sola vez. El sondeo de razones de cobertura no vistas se usa solo para calificar.

Salidas posibles por caso: clase, "indeterminado" (dentro de soporte, confianza bajo tau) o "fuera de soporte".

Uso: python pipeline/ds7_rechazo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import inferencia as I  # noqa: E402
import ds7_garantia_ml_puro as P  # noqa: E402

PRECISION_OBJETIVO = 0.95
K_VECINOS = 10
PERCENTIL_SOPORTE = 99
K_SONDEO = [0.0, 0.5, 1.5, 2.5, 4.0]


def softmax_t(logp, T):
    z = logp / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    raw = pd.read_csv(I.V2_SALIDA / P.DS / "dataset_limpio.csv", low_memory=False)
    clases = I.cargar(P.DS)["aux"]["clases"]
    X = P.construir(raw)
    y = raw["Y"].astype(int).to_numpy()
    part = raw["Particion"].astype(str).str.lower().to_numpy()
    tr, va, te = part == "train", part == "val", part == "test"

    sc = StandardScaler().fit(X.loc[tr, P.NUMERICAS])
    Xs = X.copy()
    Xs[P.NUMERICAS] = sc.transform(X[P.NUMERICAS])
    A = Xs.to_numpy(float)
    modelo = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, early_stopping=True,
                           validation_fraction=0.15, random_state=P.SEMILLA).fit(A[tr], y[tr])

    # 1. Soporte en el subespacio que define la decisión: garantía y monto (log1p, estandarizados con train).
    # Es una elección de qué espacio mide "conocido", no una regla de clasificación: la red sigue entrenada en todas las variables.
    def subespacio_df(df):
        # Garantía faltante = 0, el mismo criterio que construir() usa al rellenar la matriz de entrenamiento.
        L = np.log1p(pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in ["Valor_Garantiza", "Valor_Entregado"]})
                     .fillna(0.0).clip(lower=0))
        return L.to_numpy(float)

    S_tr_raw = subespacio_df(raw.loc[tr])
    mu_s, sd_s = S_tr_raw.mean(0), S_tr_raw.std(0)

    def subespacio(df):
        return (subespacio_df(df) - mu_s) / sd_s

    vecinos = NearestNeighbors(n_neighbors=K_VECINOS).fit((S_tr_raw - mu_s) / sd_s)

    def distancia(S):
        d, _ = vecinos.kneighbors(S)
        return d.mean(axis=1)

    d_star = float(np.percentile(distancia(subespacio(raw.loc[va])), PERCENTIL_SOPORTE))

    # 2. Calibración con temperatura en validación
    logp_va = np.log(np.clip(modelo.predict_proba(A[va]), 1e-12, None))
    yv = y[va]

    def nll(T):
        p = softmax_t(logp_va, T)
        return -np.mean(np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, None)))

    T = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
    pv = softmax_t(logp_va, T)

    # 3. Rechazo: menor tau con precisión objetivo en validación (casos dentro de soporte)
    dentro_va = distancia(subespacio(raw.loc[va])) <= d_star
    conf_va = pv.max(axis=1)
    pred_va = pv.argmax(axis=1)
    tau = None
    for t in np.arange(0.30, 0.995, 0.005):
        m = dentro_va & (conf_va >= t)
        if m.sum() >= 30 and np.mean(pred_va[m] == yv[m]) >= PRECISION_OBJETIVO:
            tau = float(t)
            break
    if tau is None:
        tau = float("nan")

    def decidir(M, df_crudo):
        dentro = distancia(subespacio(df_crudo)) <= d_star
        p = softmax_t(np.log(np.clip(modelo.predict_proba(M), 1e-12, None)), T)
        out = []
        for i in range(len(M)):
            if not dentro[i]:
                out.append("fuera de soporte")
            elif p[i].max() < tau:
                out.append("indeterminado")
            else:
                out.append(clases[int(p[i].argmax())])
        return np.array(out), p

    print(f"soporte: d* = {d_star:.3f} (percentil {PERCENTIL_SOPORTE} de validación, k = {K_VECINOS})")
    print(f"temperatura T = {T:.3f}   tau = {tau:.3f} (precisión objetivo {PRECISION_OBJETIVO:.0%} en validación)")

    # Métricas de prueba (una sola vez)
    dec_te, _ = decidir(A[te], raw.loc[te])
    yt = y[te]
    clase_real = np.array([clases[k] for k in yt])
    acept = np.isin(dec_te, clases)
    n = len(dec_te)
    print(f"\nPRUEBA ({n} casos):")
    print(f"  aceptados {acept.mean()*100:.1f} %  | precisión sobre aceptados {np.mean(dec_te[acept] == clase_real[acept])*100:.1f} %")
    print(f"  indeterminados {np.mean(dec_te == 'indeterminado')*100:.1f} %  | fuera de soporte {np.mean(dec_te == 'fuera de soporte')*100:.1f} %")
    print(f"  errores aceptados (equivocados con decisión) {np.mean(acept & (dec_te != clase_real))*100:.2f} %")

    # Sondeo de cobertura no vista
    print("\nSONDEO sobre prueba con garantía = k * Valor_Entregado (la regla solo califica):")
    print(f"{'k':>5} {'esperada':>16} {'aceptada_ok':>12} {'aceptada_mal':>13} {'indeterminado':>14} {'fuera_soporte':>14}")
    idx = np.where(te)[0]
    for k in K_SONDEO:
        R = raw.iloc[idx].copy()
        R["Valor_Garantiza"] = (k * pd.to_numeric(R["Valor_Entregado"], errors="coerce")).round(0)
        Xk = P.construir(R).reindex(columns=X.columns, fill_value=0.0)
        Xk[P.NUMERICAS] = sc.transform(Xk[P.NUMERICAS])
        dec, _ = decidir(Xk.to_numpy(float), R)
        esp = P.etiqueta_esperada(k)
        acept_k = np.isin(dec, clases)
        ok = acept_k & (dec == esp)
        mal = acept_k & (dec != esp)
        print(f"{k:5.1f} {esp:>16} {ok.mean()*100:11.1f}% {mal.mean()*100:12.1f}% "
              f"{np.mean(dec == 'indeterminado')*100:13.1f}% {np.mean(dec == 'fuera de soporte')*100:13.1f}%")


if __name__ == "__main__":
    main()
