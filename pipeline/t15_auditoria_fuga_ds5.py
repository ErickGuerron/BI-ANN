"""T15: leakage audit of the ds5 payment-timing proxies.

Compares the proxies built with three visibility rules and their univariate AUC (MORA_LEVE vs AL_DIA, train):
  A: Fecha <= Corte (the t14 rule), B: Fecha < Corte (strict), C: Fecha < first day of the cut month.
If the signal comes from payments made on or near the cut date it would drop from A to B to C. A stable AUC
means the signal comes from earlier behaviour. Also reports how many rows have payments on the cut date and
checks that the row order of dataset_limpio matches the v4 id files used by the network.
Writes resultados_keras/t15_auditoria_fuga.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t14_ds5_microatraso_viabilidad as t14  # noqa: E402


def auc_por_variante(d, F):
    y = d.Etiqueta.to_numpy()
    m = np.isin(y, ["AL_DIA", "MORA_LEVE"]) & (d.Particion.to_numpy() == "train")
    out = {}
    for c in F.columns:
        v = F.loc[m, c].fillna(F.loc[m, c].median())
        out[c] = round(float(roc_auc_score(y[m] == "MORA_LEVE", v)), 4)
    return out


def main():
    resultados = {}
    for nombre, kw in (("A_menor_igual", {}), ("B_estricto", {"estricto": True}),
                       ("C_antes_del_mes", {"excluir_mes_corte": True})):
        d, F = t14.construir(**kw)
        resultados[nombre] = {"auc": auc_por_variante(d, F), "cobertura": round(float(F.n_fechas.notna().mean()), 4)}
        print(nombre, resultados[nombre], flush=True)

    p = pd.read_parquet(t14.FACT, columns=["Num_Operacion", "Tiempo_ID"])
    p["Num_Operacion"] = p.Num_Operacion.astype(str)
    p["Fecha"] = pd.to_datetime(p.Tiempo_ID.astype(str), format="%Y%m%d")
    claves = d[["Num_Operacion", "CorteF"]].drop_duplicates()
    m = claves.merge(p[["Num_Operacion", "Fecha"]], on="Num_Operacion")
    en_corte = m[m.Fecha == m.CorteF].drop_duplicates(["Num_Operacion", "CorteF"])
    resultados["filas_con_pago_el_dia_del_corte"] = int(len(en_corte))
    resultados["filas_totales_unicas"] = int(len(claves))

    # row order versus the v4 id files
    orden = {}
    for part in ("train", "val", "test"):
        ids_v4 = pd.read_csv(t14.V4 / "ds5_pagos_multimes" / f"ids_{part}.csv").iloc[:, 0].astype(str).tolist()
        ids_d = d.loc[d.Particion == part, "Id_Fila"].astype(str).tolist()
        orden[part] = {"igual_orden": ids_v4 == ids_d, "n_v4": len(ids_v4), "n_limpio": len(ids_d)}
    resultados["orden_vs_v4"] = orden
    print(json.dumps({k: resultados[k] for k in ("filas_con_pago_el_dia_del_corte", "filas_totales_unicas",
                                                  "orden_vs_v4")}), flush=True)

    (t14.SALIDA / "t15_auditoria_fuga.json").write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
