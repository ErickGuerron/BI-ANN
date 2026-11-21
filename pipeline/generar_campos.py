"""Genera auxiliares/campos_<dataset>.json: qué campos pide el formulario, su tipo, rango observado,
si deben ser enteros o no negativos y los valores de texto conocidos en el dataset."""
import json
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import inferencia as I

TOP4 = ["ds7_historico_garantias", "ds1_ingreso_solicitudes", "ds6_gasto_provisiones", "ds5_pagos_multimes"]
AUX = I.AUX
AUX.mkdir(exist_ok=True)
for ds in sys.argv[1:] or TOP4:
    P, M = I.motor()
    cfg = M.CONFIGS[ds]
    resumen = json.loads((I.V2_SALIDA / ds / "resumen.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(Path(cfg["dir"]) / cfg["archivo"], float_precision="round_trip", low_memory=False, dtype={cfg["id"]: str})
    num, cat = resumen["columnas_numericas"], resumen["columnas_categoricas"]
    campos = []
    for c in [x for x in num + cat if x in raw.columns]:
        if c in num:
            s = pd.to_numeric(raw[c], errors="coerce").dropna()
            campos.append({"nombre": c, "tipo": "num", "min": float(s.min()), "max": float(s.max()),
                           "entero": bool(np.all(np.isclose(s, np.round(s)))) and not I.es_monetario(c),
                           "no_negativo": bool(s.min() >= 0)})
        else:
            tmp = P.limpiar_texto(raw[[cfg["id"], c]].copy(), excluir=[cfg["id"]])
            vals = sorted(tmp[c].dropna().astype(str).unique().tolist())
            campos.append({"nombre": c, "tipo": "cat", "valores": vals[:400], "n_valores": len(vals)})
    if ds == "ds7_historico_garantias" and "Valor_Garantiza" in raw.columns:
        # Network input (nucleo/garantia.py). Required: a blank is never read as 0. The user writes 0 to declare that
        # the operation has no guarantee, and the result then carries a notice.
        s = pd.to_numeric(raw["Valor_Garantiza"], errors="coerce").dropna()
        campos.append({"nombre": "Valor_Garantiza", "tipo": "num", "min": 0.0, "max": float(s.max()),
                       "entero": False, "no_negativo": True,
                       "ayuda": "Escriba 0 si la operaci\u00f3n no tiene garant\u00eda.",
                       "aviso_cero": "Garant\u00eda declarada en 0: la operaci\u00f3n se clasifica como sin garant\u00eda."})
    (AUX / f"campos_{ds}.json").write_text(json.dumps(campos, ensure_ascii=False, indent=1), encoding="utf-8")
    print(ds, len(campos), "campos;", "num:", sum(c["tipo"] == "num" for c in campos), "cat:", sum(c["tipo"] == "cat" for c in campos))
