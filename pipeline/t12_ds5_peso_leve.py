"""T12 (ds5, step a): sweep the MORA_LEVE class-weight multiplier on top of the sqrt-weighted CE.

Baseline is variant B of t4_keras (weights sqrt(N / (K n_k)), multiplier 1.0). The multiplier scales only
the MORA_LEVE weight; weights are then renormalised so the weighted mean sample weight stays 1.
The multiplier is chosen on validation macro F1 only. Test metrics are stored for every multiplier but the
selection never reads them. Writes resultados_keras/t12_ds5_peso_leve.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as t4  # noqa: E402

MULTIPLICADORES = [1.0, 1.25, 1.5, 2.0, 3.0]
CLASE_LEVE = "MORA_LEVE"


def pesos_con_multiplicador(y, K, idx_leve, m):
    w = np.sqrt(len(y) / (K * np.bincount(y, minlength=K).astype(np.float64)))
    w[idx_leve] *= m
    return (w * len(y) / w[y].sum()).astype(np.float32)


def main():
    X, y = t4.cargar_datos(t4.DS5)
    pesos, cfg = t4.cargar_config(t4.DS5)
    clases = list(pesos["clases"])
    K = len(clases)
    idx = clases.index(CLASE_LEVE)
    print("clases", clases, "idx_leve", idx, flush=True)

    resultados = {}
    for m in MULTIPLICADORES:
        filas = []
        for semilla in t4.SEMILLAS:
            w = pesos_con_multiplicador(y["train"], K, idx, m)
            modelo = t4.entrenar(X["train"], y["train"], X["val"], y["val"], K, cfg["alpha"], "wce", semilla,
                                 pesos_clase=w)
            filas.append({"semilla": semilla,
                          "val": t4.metricas(y["val"], t4.probas(modelo, X["val"])),
                          "prueba": t4.metricas(y["test"], t4.probas(modelo, X["test"]))})

        def media(split, campo):
            return round(float(np.mean([f[split][campo] for f in filas])), 4)

        def media_clase(split):
            return [round(float(np.mean([f[split]["f1_clase"][k] for f in filas])), 4) for k in range(K)]

        resultados[str(m)] = {
            "val_macro_f1": media("val", "macro_f1"),
            "val_f1_clase": media_clase("val"),
            "prueba_macro_f1": media("prueba", "macro_f1"),
            "prueba_f1_clase": media_clase("prueba"),
            "corridas": filas,
        }
        print(m, "val", resultados[str(m)]["val_macro_f1"], resultados[str(m)]["val_f1_clase"],
              "| prueba", resultados[str(m)]["prueba_macro_f1"], resultados[str(m)]["prueba_f1_clase"], flush=True)

    elegido = max(resultados, key=lambda k: resultados[k]["val_macro_f1"])
    resumen = {"clases": clases, "idx_leve": idx, "multiplicadores": MULTIPLICADORES,
               "elegido_por_validacion": elegido, "resultados": resultados}
    ruta = t4.SALIDA / "t12_ds5_peso_leve.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("elegido", elegido, "escrito", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
