import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inferencia as I
from streamlit.testing.v1 import AppTest
ds = "ds1_ingreso_solicitudes"
P, M = I.motor()
campos = json.loads((I.AUX / f"campos_{ds}.json").read_text(encoding="utf-8"))
cfg = M.CONFIGS[ds]
raw = pd.read_csv(Path(cfg["dir"]) / cfg["archivo"], float_precision="round_trip", low_memory=False, dtype={cfg["id"]: str})
fila = raw.iloc[[5]]
at = AppTest.from_file("app.py", default_timeout=170).run()
for c in campos:
    v = fila[c["nombre"]].iloc[0]
    at.text_input(key=f"{ds}_{c['nombre']}").set_value("" if pd.isna(v) else str(v))
at.run()
print("valores:", {c["nombre"]: at.text_input(key=f"{ds}_{c['nombre']}").value for c in campos[:4]})
btn = [b for b in at.button if b.label == "Clasificar operación"][1]
btn.click().run()
for m in at.markdown:
    t = m.value
    if "error" in t or "aviso" in t or "Clase" in t or "clase" in t:
        print("MD:", t[:300])
print("excepciones:", [e.value for e in at.exception])
