import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
APP = str(Path(__file__).resolve().parents[1] / "app.py")
import inferencia as I
from streamlit.testing.v1 import AppTest
TOP4 = ["ds7_historico_garantias", "ds1_ingreso_solicitudes", "ds6_gasto_provisiones", "ds5_pagos_multimes"]
P, M = I.motor()
for ds in TOP4:
    campos = json.loads((I.AUX / f"campos_{ds}.json").read_text(encoding="utf-8"))
    cfg = M.CONFIGS[ds]
    raw = pd.read_csv(Path(cfg["dir"]) / cfg["archivo"], float_precision="round_trip", low_memory=False, dtype={cfg["id"]: str})
    fila = raw.iloc[[5]]
    # ---- caso válido: valores de una fila real
    at = AppTest.from_file(APP, default_timeout=170).run()
    tab_ds = [t for t in at.tabs]
    idx = TOP4.index(ds)
    for c in campos:
        v = fila[c["nombre"]].iloc[0]
        at.text_input(key=f"{ds}_{c['nombre']}").set_value("" if pd.isna(v) else str(v))
    # el formulario del tab correspondiente: el botón de envío de cada tab
    btn = [b for b in at.button if b.label == "Clasificar operación"][idx]
    btn.click().run()
    exc = [e.value for e in at.exception]
    texto = " ".join(m.value for m in at.markdown)
    print(ds, "| válido -> excepciones:", exc, "| 'Clase predicha' visible:", "Clase predicha" in texto)
    # ---- caso inválido: campo vacío y texto en numérico
    at2 = AppTest.from_file(APP, default_timeout=170).run()
    for c in campos:
        at2.text_input(key=f"{ds}_{c['nombre']}").set_value("")
    if campos[0]["tipo"] == "num":
        at2.text_input(key=f"{ds}_{campos[0]['nombre']}").set_value("abc")
    btn2 = [b for b in at2.button if b.label == "Clasificar operación"][idx]
    btn2.click().run()
    texto2 = " ".join(m.value for m in at2.markdown)
    print("   inválido -> excepciones:", [e.value for e in at2.exception], "| errores mostrados:", texto2.count("es obligatorio") + texto2.count("escriba un número"))
