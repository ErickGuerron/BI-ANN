"""T21: run the deployed ds5 network on an external set of 100 synthetic form cases.

Usage: python pipeline/t21_ds5_casos_independientes.py <path to ANN_morosidad_100_orden_formulario_etiqueta.xlsx>

The file states that its labels are CONDITIONAL scenario labels, not observed outcomes, and asks not to compute
real accuracy. This script therefore reports agreement with the scenario label, never accuracy, and looks at
the confusion pattern only. It does not tune anything. Case 019 (empty fields) is excluded; case 003 (more
installments paid than contracted) is kept and flagged. The 32 form fields hold no payment dates, so the
payment-timing proxies cannot be evaluated here; this runs the network the app uses today (no proxies).
Writes resultados_keras/t21_casos_independientes.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROGRAMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROGRAMA / "nucleo"))
import inferencia as I  # noqa: E402
import casos_externos  # noqa: E402

DS = "ds5_pagos_multimes"
EXCLUIDOS = casos_externos.EXCLUIDOS
MARCADOS = {"MORA-NUEVO-003"}


def main(ruta):
    entrada, ids, esperado, errores = casos_externos.preparar(ruta)
    print("errores de formato segun la app:", errores or "ninguno", flush=True)
    usar = np.ones(len(ids), bool)
    clases, pr = I.predecir(DS, entrada)
    clases = np.array(clases)
    nombres = I.cargar(DS)["aux"]["clases"]
    esp, idu = esperado[usar], ids[usar]

    conf = pd.crosstab(pd.Series(esp, name="esperada"), pd.Series(clases, name="predicha"))
    conf = conf.reindex(index=nombres, columns=nombres, fill_value=0)
    print(conf.to_string(), flush=True)
    acuerdo = {c: round(float((clases[esp == c] == c).mean()), 3) for c in nombres}
    print("acuerdo con la etiqueta del escenario (NO es exactitud):", acuerdo, flush=True)

    cols_perfil = ["Dias_Desde_Ultimo_Pago", "Cuotas_Ult_3M", "Cuotas_Ult_6M", "Pct_Cuotas_Pagadas", "Saldo_Ultimo_Pago",
                   "Monto_Ult_3M", "Meses_Con_Pagos", "N_Cuotas_Con_Pago", "Numero_Cuotas", "Sin_Pagos"]
    sub = entrada.astype({c: float for c in cols_perfil if c != "Sin_Pagos"}).assign(Sin_Pagos=entrada["Sin_Pagos"].astype(float))
    perfil = {}
    for nombre, mask in (("31_90_predicho_31_90", (esp == "MORA_31_90") & (clases == "MORA_31_90")),
                         ("31_90_predicho_LEVE", (esp == "MORA_31_90") & (clases == "MORA_LEVE")),
                         ("LEVE_predicho_LEVE", (esp == "MORA_LEVE") & (clases == "MORA_LEVE")),
                         ("LEVE_predicho_31_90", (esp == "MORA_LEVE") & (clases == "MORA_31_90"))):
        if mask.sum():
            perfil[nombre] = {"n": int(mask.sum()), **{c: round(float(sub.loc[mask, c].median()), 3) for c in cols_perfil}}
    print(pd.DataFrame(perfil).T.to_string(), flush=True)

    j = {"n_usados": int(usar.sum()), "excluidos": sorted(EXCLUIDOS), "marcados": sorted(MARCADOS),
         "advertencia": "etiquetas condicionales sinteticas: acuerdo con el escenario, no exactitud",
         "confusion": {r: conf.loc[r].to_dict() for r in nombres}, "acuerdo_por_clase": acuerdo, "perfil_mediana": perfil,
         "casos": [{"id": i, "esperada": e, "predicha": p, "prob_max": round(float(m), 3),
                    "marcado": i in MARCADOS} for i, e, p, m in zip(idu, esp, clases, pr.max(1))]}
    out = PROGRAMA / "resultados_keras" / "t21_casos_independientes.json"
    out.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
