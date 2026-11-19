"""Genera resultados_convergencia/top3_<dataset>.json a partir de ajuste_<dataset>.json.

Registra las 3 mejores configuraciones de la fase 2 (ordenadas por exactitud
balanceada de CV), marca la elegida y guarda sus métricas de entrenamiento.
Las métricas de validación y prueba solo existen para la configuración elegida.
"""
import json
import sys
from pathlib import Path

CARPETA = Path(__file__).parent / "resultados_convergencia"


def construir(ajuste_path: Path) -> dict:
    d = json.loads(ajuste_path.read_text(encoding="utf-8"))
    elegida = d["config_elegida"]
    finalistas = sorted(d["finalistas"], key=lambda f: f["cv_promedio"], reverse=True)[:3]

    configuraciones = []
    for posicion, f in enumerate(finalistas, start=1):
        c = f["config"]
        es_elegida = c["capas_ocultas"] == elegida["capas_ocultas"] and c["alpha"] == elegida["alpha"]
        registro = {
            "posicion": posicion,
            "elegida": es_elegida,
            "capas_ocultas": c["capas_ocultas"],
            "neuronas_por_capa": c["capas_ocultas"],
            "numero_capas_ocultas": len(c["capas_ocultas"]),
            "alpha_l2": c["alpha"],
            "tamano_lote": c["tamano_lote"],
            "tasa_aprendizaje": c["tasa_aprendizaje"],
            "sobremuestreo": c["sobremuestreo"],
            "cv_promedio": round(f["cv_promedio"], 4),
            "cv_por_semilla": [round(s, 4) for s in f["cv_por_semilla"]],
            "epoca_mejor_cv": f["epoca"],
            "epoca_parada_cv": f["epoca_parada"],
            "validacion": None,
            "prueba": None,
            "epoca_final": None,
            "epocas_entrenadas": None,
        }
        if es_elegida:
            registro.update({
                "validacion": {k: round(v, 4) for k, v in d["validacion"].items()},
                "prueba": {k: round(v, 4) for k, v in d["prueba"].items()},
                "epoca_final": d["epoca_final"],
                "epocas_entrenadas": d["epocas_entrenadas"],
            })
        configuraciones.append(registro)

    return {
        "dataset": d["dataset"],
        "entradas": d.get("entradas"),
        "clases": d["clases"],
        "criterio": "Ordenadas por exactitud balanceada media en CV (5 folds, 4 semillas).",
        "metodo": d["metodo"],
        "receta_v4_prueba_bal": d.get("receta_v4_prueba_bal"),
        "configuraciones": configuraciones,
    }


def main(datasets):
    for ds in datasets:
        ajuste = CARPETA / f"ajuste_{ds}.json"
        salida = CARPETA / f"top3_{ds}.json"
        resultado = construir(ajuste)
        salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
        print("escrito", salida.name)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ds2_ciclo_solicitud", "ds8_notificaciones_cobranza"])
