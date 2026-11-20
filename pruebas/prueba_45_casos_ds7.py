"""Test of the ds7 network and cascade with an external file of 45 hand-built cases with sensible values that
are not in the dataset.

The file carries a CONDITIONAL expected label: a coverage rule over an assumed guarantee, which is not a model
input. Its own warning says not to compute accuracy against it. Agreement with that label is therefore reported
as indicative only; the main outputs are what the model does with plausible new cases (predicted class,
confidence, cascade state) and how many fields fall into the OTROS bucket.

Format normalization (assumption): the model vocabulary is 0/1 for the yes/no flags, 0.0/1.0 for Es_Dependiente
and "N.0" for Cargas_Familiares. The file writes SI/NO and integers, so SI -> 1 and NO -> 0 are applied. Without
this the values would fall into OTROS silently.

Usage: python pruebas/prueba_45_casos_ds7.py [path_to_csv]
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

warnings.filterwarnings("ignore")
PROGRAMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROGRAMA / "nucleo"))
import cascada as C  # noqa: E402
import inferencia as I  # noqa: E402

DS = "ds7_historico_garantias"
RUTA_DEFECTO = Path(os.environ.get("USERPROFILE", "~")) / "Downloads" / "45_casos_con_etiqueta_esperada_condicional.csv"
SALIDA = PROGRAMA / "resultados_keras"
BANDERAS_01 = ["Para_Vivienda", "Es_Agricola_Gan", "Es_Persona_Juridica"]
ANIO_MAX_ENTRENAMIENTO = 2022


def normalizar(df):
    out = df.copy()
    sino = {"SI": "1", "NO": "0"}
    for c in BANDERAS_01:
        out[c] = out[c].str.strip().str.upper().replace(sino)
    out["Es_Dependiente"] = out["Es_Dependiente"].str.strip().str.upper().replace({"SI": "1.0", "NO": "0.0"})
    out["Cargas_Familiares"] = out["Cargas_Familiares"].map(lambda v: f"{float(v):.1f}")
    return out


def campos_en_otros(df):
    """Per categorical field, how many cases hold a value outside the categories the preprocessor knows."""
    pre = I.cargar(DS)["pre"]
    res = {}
    for col, cats in pre.categorias_.items():
        if col not in df.columns:
            continue
        conocidas = set(cats[:-1])
        res[col] = int((~df[col].astype(str).str.strip().str.upper().isin(conocidas)).sum())
    return {k: v for k, v in res.items() if v}


def metricas(y, pred, K):
    return {
        "exactitud_indicativa": round(float((y == pred).mean()), 4),
        "exactitud_balanceada_indicativa": round(float(balanced_accuracy_score(y, pred)), 4),
        "macro_f1_indicativo": round(float(f1_score(y, pred, average="macro", labels=range(K), zero_division=0)), 4),
        "recall_por_clase": [round(float((pred[y == k] == k).mean()), 4) for k in range(K)],
        "matriz_confusion_filas_esperada": confusion_matrix(y, pred, labels=range(K)).tolist(),
    }


def main():
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RUTA_DEFECTO
    casos = pd.read_csv(ruta, dtype=str, encoding="utf-8")
    campos = [c["nombre"] for c in json.loads((PROGRAMA / "auxiliares" / f"campos_{DS}.json").read_text(encoding="utf-8"))]
    clases = list(I.cargar(DS)["aux"]["clases"])
    K = len(clases)
    y = casos["Y_Esperada_Condicional"].astype(int).to_numpy()

    crudo = casos[campos].copy()
    for c in ("Anio_Desembolso", "Valor_Entregado", "Numero_Cuotas", "Tasa_Interes"):
        crudo[c] = pd.to_numeric(crudo[c])
    norm = normalizar(crudo)
    otros_sin = campos_en_otros(crudo)
    otros_con = campos_en_otros(norm)
    print("campos en OTROS sin normalizar:", otros_sin)
    print("campos en OTROS tras normalizar:", otros_con, flush=True)

    entrada = I.entrada_red(DS, norm).to_numpy(float)
    P_red = I.probabilidades(DS, norm)
    pred_red = P_red.argmax(1)
    casc = C.evaluar(DS, entrada)
    pred_ens = np.array([r["clase_idx"] for r in casc])
    auto = np.array([r["estado"] == C.AUTOMATICO for r in casc])
    fuera = np.array([r["motivo"] == C.FUERA_DE_RANGO for r in casc])
    amb = np.array([r["motivo"] == C.AMBIGUEDAD for r in casc])
    nueva_epoca = (norm["Anio_Desembolso"] > ANIO_MAX_ENTRENAMIENTO).to_numpy()

    res = {
        "archivo": str(ruta.name), "n": int(len(casos)), "clases": clases,
        "advertencia": "La etiqueta esperada es condicional a una garantia supuesta que no es entrada del modelo.",
        "campos_en_otros_sin_normalizar": otros_sin, "campos_en_otros_tras_normalizar": otros_con,
        "casos_con_anio_posterior_a_2022": int(nueva_epoca.sum()),
        "red_de_la_app": {"prediccion": np.bincount(pred_red, minlength=K).tolist(),
                          "confianza_media": round(float(P_red.max(1).mean()), 4), **metricas(y, pred_red, K)},
        "ensamble": {"prediccion": np.bincount(pred_ens, minlength=K).tolist(), **metricas(y, pred_ens, K)},
        "cascada": {"automatico": round(float(auto.mean()), 4), "fuera_de_rango": round(float(fuera.mean()), 4),
                    "ambiguedad": round(float(amb.mean()), 4),
                    "automatico_por_clase_esperada": [round(float(auto[y == k].mean()), 4) for k in range(K)]},
    }
    if auto.any():
        res["cascada"]["acuerdo_indicativo_en_automaticos"] = round(float((pred_ens[auto] == y[auto]).mean()), 4)
        res["cascada"]["n_automaticos"] = int(auto.sum())
    for nombre, mascara in (("anio_hasta_2022", ~nueva_epoca), ("anio_posterior_2022", nueva_epoca)):
        if mascara.any():
            res[f"red_{nombre}"] = {"n": int(mascara.sum()), **metricas(y[mascara], pred_red[mascara], K)}
            res[f"cascada_{nombre}"] = {"automatico": round(float(auto[mascara].mean()), 4),
                                        "fuera_de_rango": round(float(fuera[mascara].mean()), 4)}

    detalle = casos[["ID_Caso", "Etiqueta_Esperada_Condicional"]].copy()
    detalle["Anio_Desembolso"] = norm["Anio_Desembolso"].to_numpy()
    detalle["Prediccion_Red"] = [clases[i] for i in pred_red]
    detalle["Confianza_Red"] = P_red.max(1).round(3)
    detalle["Prediccion_Ensamble"] = [clases[i] for i in pred_ens]
    detalle["Estado_Cascada"] = [r["estado"] for r in casc]
    detalle["Motivo"] = [r["motivo"] or "" for r in casc]
    detalle["Entropia"] = [round(r["entropia"], 3) for r in casc]
    detalle["Distancia"] = [round(r["distancia"], 2) for r in casc]
    SALIDA.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(SALIDA / "prueba_45_casos_ds7_detalle.csv", index=False, encoding="utf-8")
    (SALIDA / "prueba_45_casos_ds7.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
