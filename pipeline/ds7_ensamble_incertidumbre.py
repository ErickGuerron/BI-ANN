"""ds7 (ML puro): incertidumbre con un conjunto de redes + soporte + rechazo, sin etiquetas fuera de distribución.

Entradas: las mismas que el modelo puro (campos del formulario + Valor_Garantiza + Valor_Entregado en bruto).
Conjunto: M redes MLP con la misma arquitectura y distinta semilla.

Medidas:
  - Predicción: media de las probabilidades del conjunto. Confianza = máximo de esa media.
  - Incertidumbre epistémica: información mutua  I = H(media) - media(H(p_m)). Alta cuando las redes no coinciden.
  - Soporte: distancia media a los k vecinos de entrenamiento en el subespacio garantía-monto (log1p, estandarizado).

Rechazo (sin etiquetas de fuera de distribución):
  - Fuera de soporte: distancia > d* (percentil 99 de validación).
  - Indeterminado: I > u* (percentil 95 de I en validación dentro de soporte).
  - Aceptado: dentro de soporte y I <= u*.

Criterio de aceptación del modelo: brecha = error de prueba - error de entrenamiento del conjunto, < 3 pp.
La regla de la etiqueta solo se usa para calificar el sondeo, nunca entra al entrenamiento ni a los umbrales.

Uso: python pipeline/ds7_ensamble_incertidumbre.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import inferencia as I  # noqa: E402
import ds7_garantia_ml_puro as P  # noqa: E402

M_REDES = 5
SEMILLAS = [42, 43, 44, 45, 46]
K_VECINOS = 10
PERC_SOPORTE = 99
PERC_INCERT = 95
K_SONDEO = [0.0, 0.5, 1.5, 2.5, 4.0]


def entropia(p):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


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

    redes = []
    for s in SEMILLAS[:M_REDES]:
        redes.append(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, early_stopping=True,
                                   validation_fraction=0.15, random_state=s).fit(A[tr], y[tr]))

    def conjunto(M):
        Ps = np.stack([m.predict_proba(M) for m in redes])  # (M, n, K)
        media = Ps.mean(0)
        H_medios = -(Ps * np.log(np.clip(Ps, 1e-12, None))).sum(axis=2)  # (M, n): entropía de cada red
        I_mut = entropia(media) - H_medios.mean(0)
        return media, I_mut

    # Soporte en garantía-monto (faltante = 0, como en el entrenamiento)
    def sub_df(df):
        L = np.log1p(pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in ["Valor_Garantiza", "Valor_Entregado"]})
                     .fillna(0.0).clip(lower=0))
        return L.to_numpy(float)

    S_tr = sub_df(raw.loc[tr])
    mu, sd = S_tr.mean(0), S_tr.std(0)
    vec = NearestNeighbors(n_neighbors=K_VECINOS).fit((S_tr - mu) / sd)

    def dist_soporte(df):
        d, _ = vec.kneighbors((sub_df(df) - mu) / sd)
        return d.mean(1)

    d_star = float(np.percentile(dist_soporte(raw.loc[va]), PERC_SOPORTE))
    _, I_va = conjunto(A[va])
    dentro_va = dist_soporte(raw.loc[va]) <= d_star
    u_star = float(np.percentile(I_va[dentro_va], PERC_INCERT))

    def decidir(M, df):
        media, I_mut = conjunto(M)
        dentro = dist_soporte(df) <= d_star
        dec = []
        for i in range(len(M)):
            if not dentro[i]:
                dec.append("fuera de soporte")
            elif I_mut[i] > u_star:
                dec.append("indeterminado")
            else:
                dec.append(clases[int(media[i].argmax())])
        return np.array(dec), media, I_mut

    print(f"soporte d* = {d_star:.3f}   umbral de incertidumbre u* = {u_star:.4f} (percentil {PERC_INCERT} en validación dentro de soporte)")

    # Criterio: brecha del conjunto (predicción por media)
    def err(m):
        media, _ = conjunto(A[m])
        return 1 - float(np.mean(media.argmax(1) == y[m]))

    e_tr, e_va, e_te = err(tr), err(va), err(te)
    print(f"error conjunto: train {e_tr*100:.2f} %  val {e_va*100:.2f} %  prueba {e_te*100:.2f} %  "
          f"brecha {100*(e_te-e_tr):+.2f} pp -> {'CUMPLE' if abs(e_te-e_tr) < 0.03 else 'NO CUMPLE'} (< 3 pp)")

    # Prueba: resultado de la decisión
    dec_te, media_te, _ = decidir(A[te], raw.loc[te])
    real = np.array([clases[k] for k in y[te]])
    acep = np.isin(dec_te, clases)
    print(f"\nPRUEBA ({len(dec_te)} casos): aceptados {acep.mean()*100:.1f} %  "
          f"precisión sobre aceptados {np.mean(dec_te[acep] == real[acep])*100:.1f} %  "
          f"indeterminados {np.mean(dec_te == 'indeterminado')*100:.1f} %  "
          f"fuera de soporte {np.mean(dec_te == 'fuera de soporte')*100:.1f} %")

    # Sondeo fuera del dataset (la regla solo califica)
    print("\nSONDEO (garantía = k · monto, filas de prueba):")
    print(f"{'k':>5} {'esperada':>15} {'acept_ok':>9} {'acept_mal':>10} {'indeterm.':>10} {'fuera_sop.':>11} "
          f"{'conf acept.':>12} {'conf en error acept.':>21} {'errores atrapados':>18}")
    idx = np.where(te)[0]
    for k in K_SONDEO:
        R = raw.iloc[idx].copy()
        R["Valor_Garantiza"] = (k * pd.to_numeric(R["Valor_Entregado"], errors="coerce")).round(0)
        Xk = P.construir(R).reindex(columns=X.columns, fill_value=0.0)
        Xk[P.NUMERICAS] = sc.transform(Xk[P.NUMERICAS])
        dec, media, _ = decidir(Xk.to_numpy(float), R)
        esp = P.etiqueta_esperada(k)
        acep_k = np.isin(dec, clases)
        ok = acep_k & (dec == esp)
        mal = acep_k & (dec != esp)
        conf = media.max(1)
        rechazados = ~acep_k
        errores = ~(dec == esp)
        atrapados = (rechazados & errores).sum() / max(errores.sum(), 1)
        c_acep = conf[acep_k].mean() * 100 if acep_k.any() else float("nan")
        c_err = conf[mal].mean() * 100 if mal.any() else float("nan")
        print(f"{k:5.1f} {esp:>15} {ok.mean()*100:8.1f}% {mal.mean()*100:9.1f}% "
              f"{np.mean(dec == 'indeterminado')*100:9.1f}% {np.mean(dec == 'fuera de soporte')*100:10.1f}% "
              f"{c_acep:11.1f}% {c_err:20.1f}% {atrapados*100:17.1f}%")


if __name__ == "__main__":
    main()
