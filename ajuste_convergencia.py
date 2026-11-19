"""Ajuste por validación cruzada con parada temprana por validación (solo con TRAIN).

Cada entrenamiento se detiene cuando la exactitud balanceada de validación no mejora
más de TOL_VALIDACION durante PACIENCIA épocas. Se conserva la mejor época.

Fase 1: rejilla de arquitectura, alpha (L2) y sobremuestreo, con 5 folds estratificados y semilla 42.
Fase 2: las TOP_K mejores se repiten con SEMILLAS_REPLICA; se promedian sus curvas.
La prueba no participa en ninguna decisión; solo se reporta.

Uso:  python ajuste_convergencia.py ds2_ciclo_solicitud
Si se corta por presupuesto de tiempo, vuelve a ejecutar: retoma desde el caché (jsonl).
Salida: resultados_convergencia/
"""
import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import ann_core
from ann_core import (
    MAX_EPOCAS,
    ConfigRed,
    cargar_dataset,
    crear_modelo,
    entrenar_con_parada_temprana,
    epoca_optima,
    evaluar,
    guardar_predicciones,
    preparar_train,
    receta_original,
    tamano_lote_sugerido,
)

CARPETA_SALIDA = ann_core.CARPETA_RESULTADOS.parent / "resultados_convergencia"
REJILLA = {
    "capas_ocultas": [(32,), (64, 32), (128, 64)],
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0],
    "sobremuestreo": [False, True],
}
SEMILLA_BASE = 42
SEMILLAS_REPLICA = [1, 2, 3]
TOP_K = 3


def curva_cv(config, X, y, folds, clases, max_epocas):
    """Exactitud balanceada por época en cada fold (validación = fold no visto), promediada.
    Cada fold deja de entrenar cuando su validación se estanca; después repite su último valor."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.semilla)
    curvas = []
    for ii_tr, ii_ho in skf.split(X, y):
        X_tr, y_tr = preparar_train(config, X[ii_tr], y[ii_tr])
        modelo = crear_modelo(config)
        s = []
        for _ in range(max_epocas):
            modelo.partial_fit(X_tr, y_tr, classes=clases)
            s.append(balanced_accuracy_score(y[ii_ho], modelo.predict(X[ii_ho])))
            if len(s) >= ann_core.PACIENCIA and ann_core.epoca_de_plateau([-v for v in s]) is not None:
                break
        s += [s[-1]] * (max_epocas - len(s))
        curvas.append(s)
    return np.mean(curvas, axis=0)


def resumen(curva):
    mejor, parada = epoca_optima(curva.tolist())
    return {"epoca": mejor, "epoca_parada": parada, "cv": float(curva[mejor - 1]), "curva": curva.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epocas", type=int, default=MAX_EPOCAS)
    parser.add_argument("--presupuesto", type=float, default=150, help="segundos de cómputo por ejecución")
    args = parser.parse_args()
    inicio_ejecucion = time.time()

    datos = cargar_dataset(args.dataset)
    train, val, test = datos["train"], datos["val"], datos["test"]
    clases = np.unique(train["y"])
    lote = tamano_lote_sugerido(len(train["y"]))
    CARPETA_SALIDA.mkdir(exist_ok=True)
    ruta_cache = CARPETA_SALIDA / f"cache_{args.dataset}.jsonl"
    cache = {}
    if ruta_cache.exists():
        for linea in ruta_cache.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                fila = json.loads(linea)
                cache[json.dumps(fila["config"], sort_keys=True)] = fila["resultado"]

    def config_con(capas, alpha, sobre, semilla):
        return ConfigRed(capas_ocultas=capas, alpha=alpha, tamano_lote=lote, sobremuestreo=sobre, semilla=semilla)

    def obtener(config):
        clave = json.dumps(config.a_dict(), sort_keys=True)
        if clave in cache:
            return cache[clave]
        if time.time() - inicio_ejecucion > args.presupuesto:
            return None
        inicio = time.time()
        res = resumen(curva_cv(config, train["X"], train["y"], args.folds, clases, args.max_epocas))
        with ruta_cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"config": config.a_dict(), "resultado": res}) + "\n")
        cache[clave] = res
        print(f"[cache] capas={config.capas_ocultas} alpha={config.alpha} sobre={config.sobremuestreo} "
              f"semilla={config.semilla} -> cv={res['cv']:.4f} epoca={res['epoca']} parada={res['epoca_parada']} "
              f"({round(time.time() - inicio, 1)}s)", flush=True)
        return res

    # Fase 1: rejilla completa con la semilla base
    bases = [config_con(c, a, s, SEMILLA_BASE) for c, a, s in
             itertools.product(REJILLA["capas_ocultas"], REJILLA["alpha"], REJILLA["sobremuestreo"])]
    res_base = []
    for i, cfg in enumerate(bases):
        r = obtener(cfg)
        if r is None:
            print(f"PENDIENTE fase 1: {i}/{len(bases)} listas. Vuelve a ejecutar.", flush=True)
            return
        res_base.append(r)
    tabla_fase1 = [{"config": c.a_dict(), "epoca": r["epoca"], "cv": r["cv"]} for c, r in zip(bases, res_base)]
    orden = sorted(range(len(bases)), key=lambda i: res_base[i]["cv"], reverse=True)

    # Fase 2: las TOP_K mejores con otras semillas; se promedian las curvas de validación
    finalistas = []
    for i in orden[:TOP_K]:
        base = bases[i]
        curvas = [np.array(res_base[i]["curva"])]
        for s in SEMILLAS_REPLICA:
            r = obtener(config_con(base.capas_ocultas, base.alpha, base.sobremuestreo, s))
            if r is None:
                print("PENDIENTE fase 2. Vuelve a ejecutar.", flush=True)
                return
            curvas.append(np.array(r["curva"]))
        res = resumen(np.mean(curvas, axis=0))
        finalistas.append({
            "config": base.a_dict(),
            "config_obj": base,
            "epoca": res["epoca"],
            "epoca_parada": res["epoca_parada"],
            "cv_promedio": res["cv"],
            "cv_por_semilla": [float(c[res["epoca"] - 1]) for c in curvas],
        })
    mejor = max(finalistas, key=lambda f: f["cv_promedio"])
    config_final = mejor["config_obj"]

    modelo, historial, epoca_final = entrenar_con_parada_temprana(config_final, train, val, args.max_epocas)
    res_val = evaluar(modelo, val["X"], val["y"])
    res_test = evaluar(modelo, test["X"], test["y"])
    guardar_predicciones(modelo, test, datos["clases"], CARPETA_SALIDA / f"predicciones_prueba_{args.dataset}.csv")

    entradas = list(pd.read_csv(ann_core.CARPETA_DATOS / args.dataset / "X_train_nn.csv", nrows=0).columns)
    pesos = {
        "dataset": args.dataset,
        "clases": datos["clases"],
        "entradas": entradas,
        "capas": [len(entradas)] + list(config_final.capas_ocultas) + [len(datos["clases"])],
        "coefs": [w.tolist() for w in modelo.coefs_],
        "intercepts": [b.tolist() for b in modelo.intercepts_],
        "config": config_final.a_dict(),
        "epoca_final": epoca_final,
    }
    (CARPETA_SALIDA / f"pesos_{args.dataset}.json").write_text(json.dumps(pesos, ensure_ascii=False), encoding="utf-8")

    salida = {
        "dataset": args.dataset,
        "clases": datos["clases"],
        "metodo": (f"Parada temprana: se detiene cuando la exactitud balanceada de validación no mejora más de "
                   f"{ann_core.TOL_VALIDACION} durante {ann_core.PACIENCIA} épocas seguidas (máximo {args.max_epocas}), "
                   f"y se conserva la mejor época. Fase 1: CV estratificada {args.folds} folds sobre train, semilla "
                   f"{SEMILLA_BASE}. Fase 2: top {TOP_K} repetidas con semillas {SEMILLAS_REPLICA}. Sobremuestreo solo en train."),
        "config_elegida": config_final.a_dict(),
        "epoca_final": epoca_final,
        "epocas_entrenadas": len(historial),
        "epoca_cv": mejor["epoca"],
        "cv_promedio": mejor["cv_promedio"],
        "cv_por_semilla": mejor["cv_por_semilla"],
        "finalistas": [{k: v for k, v in f.items() if k != "config_obj"} for f in finalistas],
        "validacion": res_val,
        "prueba": res_test,
        "receta_v4_prueba_bal": receta_original(args.dataset).get("bal_prueba"),
        "historial": historial,
        "tabla_fase1": tabla_fase1,
    }
    ruta = CARPETA_SALIDA / f"ajuste_{args.dataset}.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULTADO", json.dumps({"config": config_final.a_dict(), "epoca_final": epoca_final,
                                  "cv": mejor["cv_promedio"], "val": res_val, "prueba": res_test}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
