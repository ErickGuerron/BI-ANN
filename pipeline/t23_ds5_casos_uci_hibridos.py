"""T23: run the deployed ds5 network on UCI_100_formulario_COMPLETO_SINTETICO.xlsx (hybrid synthetic cases).

The file says: demographics come from UCI Default of Credit Card Clients (only age band, sex, civil status and
education where compatible), finances are synthetic, built from ds5 cooperative profiles and modified, and the
label is a conditional proxy. It also says it does NOT measure real accuracy or rule out memorisation, and is meant
for manual, integration, UI and robustness tests. This script reports agreement with the conditional label and
where the reference profile of each case sits (train / validation / test of ds5). Cases go through app.revisar
(see casos_externos.py). Nothing is tuned. Usage: python pipeline/t23_ds5_casos_uci_hibridos.py <xlsx>.
Writes resultados_keras/t23_casos_uci_hibridos.json.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inferencia as I  # noqa: E402
import casos_externos  # noqa: E402

DS = "ds5_pagos_multimes"
LIMPIO = PROGRAMA.parents[1] / "datasets_multiclase_v2/salida/ds5_pagos_multimes/dataset_limpio.csv"


def main(ruta):
    casos_externos.EXCLUIDOS = set()
    entrada, ids, esperado, errores = casos_externos.preparar(ruta)
    print("casos:", len(ids), "| errores de formato segun la app:", errores or "ninguno", flush=True)
    clases, pr = I.predecir(DS, entrada)
    clases = np.array(clases)
    nombres = I.cargar(DS)["aux"]["clases"]

    conf = pd.crosstab(pd.Series(esperado, name="esperada"), pd.Series(clases, name="predicha"))
    conf = conf.reindex(index=nombres, columns=nombres, fill_value=0)
    print(conf.to_string(), flush=True)
    acuerdo = {c: round(float((clases[esperado == c] == c).mean()), 3) for c in nombres}
    print("acuerdo con la etiqueta condicional (NO es exactitud):", acuerdo,
          "| global", round(float((clases == esperado).mean()), 3), flush=True)

    tr = pd.read_excel(ruta, sheet_name="Trazabilidad")
    part = pd.read_csv(LIMPIO, usecols=["Id_Fila", "Particion", "Etiqueta"]).set_index("Id_Fila")
    ref = tr["ID_Perfil_Financiero_Referencia"].astype(str)
    particion = ref.map(part["Particion"]).fillna("no encontrada").to_numpy()
    en_dataset = (particion != "no encontrada")
    print("perfil de referencia por particion de ds5:", pd.Series(particion).value_counts().to_dict(), flush=True)
    por_part = {}
    for p in ("train", "val", "test"):
        m = particion == p
        if m.sum():
            por_part[p] = {"n": int(m.sum()), "acuerdo": round(float((clases[m] == esperado[m]).mean()), 3)}
    print("acuerdo segun la particion del perfil de referencia:", por_part, flush=True)

    j = {"n": len(ids), "advertencia": "etiquetas condicionales sinteticas e hibridas: acuerdo, no exactitud",
         "errores_de_formato": [list(e) for e in errores],
         "confusion": {r: conf.loc[r].to_dict() for r in nombres}, "acuerdo_por_clase": acuerdo,
         "acuerdo_global": round(float((clases == esperado).mean()), 3),
         "particion_del_perfil_de_referencia": pd.Series(particion).value_counts().to_dict(),
         "acuerdo_por_particion_de_referencia": por_part,
         "casos": [{"id": i, "esperada": e, "predicha": p, "prob_max": round(float(m), 3), "ref": r, "particion_ref": q}
                   for i, e, p, m, r, q in zip(ids, esperado, clases, pr.max(1), ref, particion)]}
    (PROGRAMA / "resultados_keras" / "t23_casos_uci_hibridos.json").write_text(
        json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
