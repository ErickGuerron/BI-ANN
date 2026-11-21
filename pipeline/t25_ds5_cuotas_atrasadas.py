"""T25 (ds5, step 3): installment-lag signal = installments due by the cut date minus installments paid.

Installments due come from FACT_Desembolsos.Fecha_Desembolso (monthly schedule assumed, checked against the
payment calendar) and Numero_Cuotas. Installments paid use only payments with Fecha < Corte. Univariate check on
train + val only (test never read): AUC for MORA_LEVE vs AL_DIA and MORA_31_90 vs MORA_LEVE, plus the class
medians of each signal. Writes resultados_keras/t25_cuotas_atrasadas.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402

DESEMBOLSOS = t14.FACT.parents[1] / "FACT_Desembolsos/FACT_Desembolsos_datos.parquet"
CLASES = ["AL_DIA", "MORA_LEVE", "MORA_31_90", "MORA_MAYOR_90"]


def senales(d):
    """Return DataFrame of lag signals aligned with d (needs Num_Operacion, CorteF, Numero_Cuotas)."""
    des = pd.read_parquet(DESEMBOLSOS, columns=["Num_Operacion", "Fecha_Desembolso"])
    des["Num_Operacion"] = des.Num_Operacion.astype(str)
    des["Fecha_Desembolso"] = pd.to_datetime(des.Fecha_Desembolso)
    des = des.drop_duplicates("Num_Operacion").set_index("Num_Operacion").Fecha_Desembolso
    p = pd.read_parquet(t14.FACT, columns=["Num_Operacion", "Num_Cuota", "Tiempo_ID"])
    p["Fecha"] = pd.to_datetime(p.Tiempo_ID.astype(str), format="%Y%m%d")
    p["Num_Operacion"] = p.Num_Operacion.astype(str)
    grupos = {k: g[["Fecha", "Num_Cuota"]] for k, g in p.groupby("Num_Operacion")}
    filas = []
    for op, corte, n in zip(d.Num_Operacion, d.CorteF, d.Numero_Cuotas):
        f0 = des.get(op, pd.NaT)
        if pd.isna(f0):
            filas.append({})
            continue
        meses = (corte.year - f0.year) * 12 + (corte.month - f0.month) - (1 if corte.day < f0.day else 0)
        esperadas = float(min(max(meses, 0), n))
        g = grupos.get(op)
        pagadas = 0.0
        if g is not None:
            v = g[g.Fecha < corte]
            pagadas = float(v.Num_Cuota.nunique()) if not v.empty else 0.0
        mx = float(v.Num_Cuota.max()) if g is not None and not v.empty else 0.0
        filas.append({"meses_desde_desemb": float(meses), "cuotas_esperadas": esperadas,
                      "cuotas_atrasadas": esperadas - pagadas, "cuotas_atrasadas_max": esperadas - mx,
                      "ratio_atraso": (esperadas - pagadas) / max(esperadas, 1.0)})
    return pd.DataFrame(filas, index=d.index)


def main():
    d, _ = t14.construir(estricto=True)
    S = senales(d)
    cobertura = float(S.cuotas_atrasadas.notna().mean())
    y = d.Etiqueta.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    sel = np.isin(d.Particion.to_numpy(), ["train", "val"])
    res = {"cobertura": round(cobertura, 4), "auc_leve_vs_aldia": {}, "auc_3190_vs_leve": {}, "mediana_por_clase": {}}
    for c in S.columns:
        v = S[c].fillna(S[c].median())
        for clave, (a, b) in (("auc_leve_vs_aldia", (0, 1)), ("auc_3190_vs_leve", (1, 2))):
            m = sel & np.isin(y, [a, b])
            res[clave][c] = round(float(roc_auc_score(y[m] == b, v[m])), 4)
        res["mediana_por_clase"][c] = {CLASES[k]: round(float(S.loc[sel & (y == k), c].median()), 2) for k in range(4)}
    print(json.dumps(res, indent=1))
    t14.SALIDA.mkdir(exist_ok=True)
    (t14.SALIDA / "t25_cuotas_atrasadas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    S.to_parquet(t14.SALIDA / "_t25_senales.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
