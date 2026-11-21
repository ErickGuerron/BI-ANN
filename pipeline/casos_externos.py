"""Prepares an external case file exactly as the app does: every field goes through app.revisar, so a typed
`15` becomes the canonical category `15.0`, SI/NO become the dataset codes, and so on. Bypassing this step
makes the network read known categories as OTROS (that happened in the first version of t21/t22).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
PROGRAMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROGRAMA))
sys.path.insert(0, str(PROGRAMA / "nucleo"))
import app  # noqa: E402

DS = "ds5_pagos_multimes"
EXCLUIDOS = {"MORA-NUEVO-019"}  # empty fields in the source file


def _bruto(c, v):
    """The value as the user would give it to the widget of that field."""
    if pd.isna(v):
        return ""
    if app.es_booleano(c):
        return "SI" if str(v).strip().upper() in ("SÍ", "SI", "1", "1.0") else "NO"
    if c["tipo"] == "num":
        return float(v)
    return str(v)


def preparar(ruta, ds=DS):
    """Returns (cleaned DataFrame, ids, expected labels, list of (id, field, error))."""
    d = pd.read_excel(ruta, sheet_name=0)
    d = d[~d["ID_Caso"].isin(EXCLUIDOS)].reset_index(drop=True)
    campos = app.cargar_campos(ds)
    filas, errores = [], []
    for _, fila in d.iterrows():
        limpio = {}
        for c in campos:
            if c["nombre"] not in d.columns:
                continue
            err, _, valor = app.revisar(c, _bruto(c, fila[c["nombre"]]))
            if err:
                errores.append((fila["ID_Caso"], c["nombre"], err))
            limpio[c["nombre"]] = valor
        filas.append(limpio)
    return (pd.DataFrame(filas), d["ID_Caso"].to_numpy(), d["Etiqueta_Esperada_Condicional"].to_numpy(), errores)
