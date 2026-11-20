"""End-to-end simulation of the ds7 cascade through the app path: raw record -> inferencia.entrada_red -> cascada.

1. Parity: the app's standardized input must equal X_test_nn.csv for the same operations (the cascade and its
   k-NN filter live in that space).
2. Real test records: share automatic / out of range / ambiguous, and accuracy of the automatic ones.
3. Perturbed raw fields (values no record in the data has): moderate and extreme.
Run: python pruebas/simulacion_app_ds7.py   (needs auxiliares/cascada_ds7_historico_garantias.*)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROGRAMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROGRAMA / "nucleo"))
import cascada as C  # noqa: E402
import inferencia as I  # noqa: E402

DS = "ds7_historico_garantias"
BRUTO = PROGRAMA.parents[1] / "para_companero_cadena_datasets" / "datasets_bayes" / "datasets" / f"{DS}.csv"
N = 600
SEMILLA = 5


def resumen(res):
    n = len(res)
    est = [r["estado"] for r in res]
    mot = [r["motivo"] for r in res]
    return {"automatico": round(est.count(C.AUTOMATICO) / n, 4),
            "fuera_de_rango": round(mot.count(C.FUERA_DE_RANGO) / n, 4),
            "ambiguedad": round(mot.count(C.AMBIGUEDAD) / n, 4)}


def main():
    campos = [c["nombre"] for c in json.loads((PROGRAMA / "auxiliares" / f"campos_{DS}.json").read_text(encoding="utf-8"))]
    raw = pd.read_csv(BRUTO)
    raw["Num_Operacion"] = raw["Num_Operacion"].astype(str)
    ids = pd.read_csv(PROGRAMA.parent / DS / "ids_test.csv")["Num_Operacion"].astype(str).to_numpy()
    Xnn = pd.read_csv(PROGRAMA.parent / DS / "X_test_nn.csv").to_numpy(float)
    y = pd.read_csv(PROGRAMA.parent / DS / "y_test.csv")["Y"].to_numpy().astype(int)

    rng = np.random.default_rng(SEMILLA)
    pos = np.sort(rng.choice(len(ids), size=N, replace=False))
    base = raw.set_index("Num_Operacion").loc[ids[pos], campos].reset_index(drop=True)

    entrada = I.entrada_red(DS, base).to_numpy(float)
    paridad = float(np.abs(entrada - Xnn[pos]).max())
    print("paridad max|entrada_red - X_test_nn|", paridad, flush=True)

    out = {"paridad_max_abs": paridad, "n": N}
    res = C.evaluar(DS, entrada)
    auto = np.array([r["estado"] == C.AUTOMATICO for r in res])
    pred = np.array([r["clase_idx"] for r in res])
    out["registros_reales"] = {**resumen(res),
                               "exactitud_automaticos": round(float((pred[auto] == y[pos][auto]).mean()), 4)}
    print("registros reales", out["registros_reales"], flush=True)

    def perturbar(nombre, fn):
        df = base.copy()
        fn(df)
        r = C.evaluar(DS, I.entrada_red(DS, df).to_numpy(float))
        out[nombre] = resumen(r)
        print(nombre, out[nombre], flush=True)

    perturbar("valor_entregado_x3", lambda d: d.__setitem__("Valor_Entregado", d["Valor_Entregado"] * 3))
    perturbar("valor_entregado_x50", lambda d: d.__setitem__("Valor_Entregado", d["Valor_Entregado"] * 50))
    perturbar("numero_cuotas_x5", lambda d: d.__setitem__("Numero_Cuotas", d["Numero_Cuotas"] * 5))
    perturbar("anio_9999", lambda d: d.__setitem__("Anio_Desembolso", 9999))
    perturbar("tasa_99", lambda d: d.__setitem__("Tasa_Interes", 99))
    perturbar("canton_nuevo", lambda d: d.__setitem__("Canton", "CANTON NUEVO"))
    perturbar("producto_y_sucursal_nuevos", lambda d: (d.__setitem__("Nombre_Producto", "PRODUCTO NUEVO"),
                                                       d.__setitem__("Sucursal", "SUCURSAL NUEVA")))
    for nombre in ("alto", "bajo"):
        fila = json.loads((PROGRAMA / "pruebas" / f"_tmp_{DS}_{nombre}.json").read_text(encoding="utf-8"))
        r = C.evaluar(DS, I.entrada_red(DS, pd.DataFrame([fila])[campos]).to_numpy(float))
        out[f"ejemplo_{nombre}"] = {"estado": r[0]["estado"], "motivo": r[0]["motivo"],
                                    "entropia": round(r[0]["entropia"], 3), "distancia": round(r[0]["distancia"], 3)}
        print("ejemplo", nombre, out[f"ejemplo_{nombre}"], flush=True)

    ruta = PROGRAMA / "resultados_keras" / "simulacion_app_ds7.json"
    ruta.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
