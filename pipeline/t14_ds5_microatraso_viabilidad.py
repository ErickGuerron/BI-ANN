"""T14 (ds5, step c): feasibility check of micro-delay features derived from the payment-level fact table.

Source facts (verified):
  - FACT_Pagos_Y_Recaudaciones_muestra has payment dates (Tiempo_ID) and Num_Cuota per movement, but
    Dias_Atraso_Al_Pago is 0 in every row, so no real per-event delay exists.
  - The ds5 label is Info_Dias_Morosidad_Objetivo bucketed: MORA_LEVE = 1..30 days.
This script derives timing proxies from payment dates strictly up to each cut date (Corte), then checks, on
train and validation only (test is never read):
  1. univariate AUC of each proxy, MORA_LEVE vs AL_DIA;
  2. HistGradientBoosting on the existing numeric inputs with and without the proxies, MORA_LEVE F1 and
     macro F1 on validation, 3 seeds.
Writes resultados_keras/t14_microatraso.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

PROGRAMA = Path(__file__).resolve().parents[1]
V4 = PROGRAMA.parent
BI = V4.parent
FACT = BI / "proyecto_pedrodw/proyecto/salidas/FACT_Pagos_Y_Recaudaciones_muestra/FACT_Pagos_Y_Recaudaciones_muestra_datos.parquet"
LIMPIO = BI / "datasets_multiclase_v2/salida/ds5_pagos_multimes/dataset_limpio.csv"
SALIDA = PROGRAMA / "resultados_keras"
VENTANA_DIAS = 183


def proxies(fechas_cuotas, corte):
    """fechas_cuotas: DataFrame(Fecha, Num_Cuota) of one operation, Fecha <= corte."""
    nulo = {k: np.nan for k in ("gap_max", "gap_medio", "gap_desv", "n_fechas", "dia_ult", "dia_ult_menos_med",
                                 "dia_desv", "frac_cuotas_parciales", "dias_ult_pago_vs_gap_medio")}
    f = fechas_cuotas[fechas_cuotas.Fecha > corte - pd.Timedelta(days=VENTANA_DIAS)]
    if f.empty:
        return nulo
    fechas = np.sort(f.Fecha.unique())
    gaps = np.diff(fechas).astype("timedelta64[D]").astype(float) if len(fechas) > 1 else np.array([])
    dias = pd.DatetimeIndex(fechas).day.to_numpy().astype(float)
    por_cuota = f.groupby("Num_Cuota").Fecha.nunique()
    ult = pd.Timestamp(fechas[-1])
    gm = gaps.mean() if len(gaps) else np.nan
    return {
        "gap_max": gaps.max() if len(gaps) else np.nan,
        "gap_medio": gm,
        "gap_desv": gaps.std() if len(gaps) > 1 else np.nan,
        "n_fechas": float(len(fechas)),
        "dia_ult": dias[-1],
        "dia_ult_menos_med": dias[-1] - np.median(dias),
        "dia_desv": dias.std(),
        "frac_cuotas_parciales": float((por_cuota > 1).mean()),
        "dias_ult_pago_vs_gap_medio": (corte - ult).days - gm if not np.isnan(gm) else np.nan,
    }


def construir():
    d = pd.read_csv(LIMPIO)
    d["CorteF"] = pd.to_datetime(d.Corte.astype(str), format="%Y%m%d")
    p = pd.read_parquet(FACT, columns=["Num_Operacion", "Num_Cuota", "Tiempo_ID"])
    p["Fecha"] = pd.to_datetime(p.Tiempo_ID.astype(str), format="%Y%m%d")
    p["Num_Operacion"] = p.Num_Operacion.astype(str)
    d["Num_Operacion"] = d.Num_Operacion.astype(str)
    grupos = {k: g[["Fecha", "Num_Cuota"]] for k, g in p.groupby("Num_Operacion")}
    vacio = pd.DataFrame({"Fecha": pd.Series([], dtype="datetime64[ns]"), "Num_Cuota": pd.Series([], dtype="int64")})
    filas = []
    for op, corte in zip(d.Num_Operacion, d.CorteF):
        g = grupos.get(op)
        filas.append(proxies(g[g.Fecha <= corte], corte) if g is not None else proxies(vacio, corte))
    return d, pd.DataFrame(filas, index=d.index)


def main():
    d, F = construir()
    cobertura = float(F.n_fechas.notna().mean())
    print("cobertura de filas con pagos en ventana:", round(cobertura, 4), flush=True)
    clases = ["AL_DIA", "MORA_LEVE", "MORA_31_90", "MORA_MAYOR_90"]
    y = d.Etiqueta.map({c: i for i, c in enumerate(clases)}).to_numpy()
    sel = d.Particion.to_numpy()

    m = np.isin(y, [0, 1]) & (sel == "train")
    auc = {}
    for c in F.columns:
        v = F.loc[m, c].fillna(F.loc[m, c].median())
        auc[c] = round(float(roc_auc_score(y[m] == 1, v)), 4)
    print("AUC univariado MORA_LEVE vs AL_DIA (train):", auc, flush=True)

    base_cols = [c for c in ["Valor_Entregado", "Numero_Cuotas", "N_Movimientos", "Ultima_Cuota_Pagada",
                              "Monto_Total_Cobrado", "Monto_Promedio_Mov", "Cuotas_Ult_3M", "Cuotas_Ult_6M",
                              "Monto_Ult_3M", "Monto_Ult_6M", "Saldo_Ultimo_Pago", "Dias_Desde_Ultimo_Pago",
                              "Meses_Con_Pagos", "Pct_Cuotas_Pagadas", "Pct_Saldo_Ultimo_Pago",
                              "Ratio_Monto_3M_6M", "Tasa_Interes", "Sin_Pagos"] if c in d.columns]
    Xb = d[base_cols].astype(float)
    Xn = pd.concat([Xb, F], axis=1)
    res = {}
    for nombre, X in (("base", Xb), ("base+micro", Xn)):
        filas = []
        for semilla in (42, 43, 44):
            mod = HistGradientBoostingClassifier(random_state=semilla, class_weight="balanced", max_iter=300,
                                                  learning_rate=0.05, early_stopping=True)
            mod.fit(X[sel == "train"], y[sel == "train"])
            pred = mod.predict(X[sel == "val"])
            yv = y[sel == "val"]
            f = f1_score(yv, pred, average=None, labels=range(4), zero_division=0)
            filas.append({"macro_f1": float(f.mean()), "f1_leve": float(f[1]), "f1_clase": f.round(4).tolist()})
        res[nombre] = {"macro_f1": round(float(np.mean([r["macro_f1"] for r in filas])), 4),
                       "f1_leve": round(float(np.mean([r["f1_leve"] for r in filas])), 4),
                       "corridas": filas}
        print(nombre, res[nombre]["macro_f1"], res[nombre]["f1_leve"], flush=True)

    SALIDA.mkdir(exist_ok=True)
    (SALIDA / "t14_microatraso.json").write_text(json.dumps(
        {"cobertura": cobertura, "auc_univariado": auc, "hgb_validacion": res}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
