"""Ajuste de la red por validación cruzada (solo con TRAIN), en dos fases.

Fase 1: rejilla de arquitectura, alpha (L2) y sobremuestreo, con 5 folds estratificados y semilla 42.
Fase 2: las TOP_K mejores combinaciones se repiten con SEMILLAS_REPLICA (otros folds e inicializaciones).
        La decisión usa el promedio de las curvas de todas las semillas, para ver si la mejora se sostiene.
Época óptima: la primera época cuya exactitud balanceada CV queda a TOLERANCIA o menos del máximo.
La prueba no participa en ninguna decisión; solo se reporta.

Uso:  python ajuste.py ds8_notificaciones_cobranza
Si se corta por tiempo (presupuesto de la ejecución), vuelve a ejecutar: retoma desde el caché.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))

from ann_core import (  # noqa: E402
    CARPETA_RESULTADOS,
    ConfigRed,
    cargar_dataset,
    crear_modelo,
    entrenar,
    evaluar,
    guardar_predicciones,
    preparar_train,
    receta_original,
    tamano_lote_sugerido,
)

REJILLA = {
    "capas_ocultas": [(32,), (64, 32), (128, 64)],
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0],
    "sobremuestreo": [False, True],
}
SEMILLA_BASE = 42
SEMILLAS_REPLICA = [1, 2, 3]
TOP_K = 3
TOLERANCIA = 0.005


def curva_validacion_cruzada(config, X, y, max_epocas, folds, clases):
    """Exactitud balanceada por época, promediada sobre los folds (cada fold valida con datos no vistos)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.semilla)
    curvas = np.zeros((folds, max_epocas))
    for k, (ii_tr, ii_ho) in enumerate(skf.split(X, y)):
        X_tr, y_tr = preparar_train(config, X[ii_tr], y[ii_tr])
        modelo = crear_modelo(config)
        for ep in range(max_epocas):
            modelo.partial_fit(X_tr, y_tr, classes=clases)
            curvas[k, ep] = balanced_accuracy_score(y[ii_ho], modelo.predict(X[ii_ho]))
    return curvas.mean(axis=0)


def elegir_epoca(curva):
    """Primera época cuyo score queda a TOLERANCIA o menos del máximo de la curva."""
    return int(np.flatnonzero(curva >= curva.max() - TOLERANCIA)[0]) + 1


def leer_cache(ruta):
    cache = {}
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                fila = json.loads(linea)
                cache[json.dumps(fila["config"], sort_keys=True)] = np.array(fila["curva"])
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epocas", type=int, default=60)
    parser.add_argument("--presupuesto", type=float, default=150, help="segundos de cómputo por ejecución")
    args = parser.parse_args()
    inicio_ejecucion = time.time()

    datos = cargar_dataset(args.dataset)
    train, val, test = datos["train"], datos["val"], datos["test"]
    clases = np.unique(train["y"])
    lote = tamano_lote_sugerido(len(train["y"]))
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    ruta_cache = CARPETA_RESULTADOS / f"curvas_{args.dataset}.jsonl"
    cache = leer_cache(ruta_cache)

    def obtener_curva(config):
        clave = json.dumps(config.a_dict(), sort_keys=True)
        if clave in cache:
            return cache[clave]
        if time.time() - inicio_ejecucion > args.presupuesto:
            return None
        inicio = time.time()
        curva = curva_validacion_cruzada(config, train["X"], train["y"], args.max_epocas, args.folds, clases)
        with ruta_cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"config": config.a_dict(), "curva": curva.tolist()}) + "\n")
        cache[clave] = curva
        ep = elegir_epoca(curva)
        print(f"[fase cache] capas={config.capas_ocultas} alpha={config.alpha} sobre={config.sobremuestreo} "
              f"semilla={config.semilla} -> cv={curva[ep-1]:.4f} epoca={ep} ({round(time.time()-inicio,1)}s)", flush=True)
        return curva

    def config_con(capas, alpha, sobre, semilla):
        return ConfigRed(capas_ocultas=capas, alpha=alpha, tamano_lote=lote, sobremuestreo=sobre, semilla=semilla)

    # Fase 1: rejilla completa con la semilla base
    bases = [config_con(c, a, s, SEMILLA_BASE) for c, a, s in
             itertools.product(REJILLA["capas_ocultas"], REJILLA["alpha"], REJILLA["sobremuestreo"])]
    curvas_base = []
    for cfg in bases:
        curva = obtener_curva(cfg)
        if curva is None:
            print(f"PENDIENTE fase 1: {len([c for c in curvas_base])}/{len(bases)} listas. Vuelve a ejecutar.", flush=True)
            return
        curvas_base.append(curva)
    tabla_fase1 = []
    for cfg, curva in zip(bases, curvas_base):
        ep = elegir_epoca(curva)
        tabla_fase1.append({"config": cfg.a_dict(), "epoca": ep, "cv": float(curva[ep - 1])})
    orden = sorted(range(len(bases)), key=lambda i: tabla_fase1[i]["cv"], reverse=True)

    # Fase 2: repetir las TOP_K mejores con otras semillas
    finalistas = []
    for i in orden[:TOP_K]:
        curvas = [curvas_base[i]]
        for s in SEMILLAS_REPLICA:
            base = bases[i]
            curva = obtener_curva(config_con(base.capas_ocultas, base.alpha, base.sobremuestreo, s))
            if curva is None:
                print("PENDIENTE fase 2. Vuelve a ejecutar.", flush=True)
                return
            curvas.append(curva)
        promedio = np.mean(curvas, axis=0)
        ep = elegir_epoca(promedio)
        finalistas.append({
            "config": bases[i].a_dict(),
            "config_obj": bases[i],
            "epoca": ep,
            "cv_promedio": float(promedio[ep - 1]),
            "cv_por_semilla": [float(c[ep - 1]) for c in curvas],
        })
    mejor = max(finalistas, key=lambda f: f["cv_promedio"])
    config_final, epocas = mejor["config_obj"], mejor["epoca"]

    modelo, historial = entrenar(config_final, train, val, epocas)
    res_val = evaluar(modelo, val["X"], val["y"])
    res_test = evaluar(modelo, test["X"], test["y"])
    guardar_predicciones(modelo, test, datos["clases"], CARPETA_RESULTADOS / f"predicciones_prueba_{args.dataset}.csv")

    salida = {
        "dataset": args.dataset,
        "clases": datos["clases"],
        "metodo": (f"Fase 1: CV estratificada {args.folds} folds sobre train, semilla {SEMILLA_BASE}. "
                   f"Fase 2: top {TOP_K} repetidas con semillas {SEMILLAS_REPLICA}. "
                   f"Época = primera a {TOLERANCIA} del máximo de la curva promedio. Sobremuestreo solo en train."),
        "config_elegida": config_final.a_dict(),
        "epocas": epocas,
        "cv_promedio": mejor["cv_promedio"],
        "cv_por_semilla": mejor["cv_por_semilla"],
        "finalistas": [{k: v for k, v in f.items() if k != "config_obj"} for f in finalistas],
        "validacion": res_val,
        "prueba": res_test,
        "receta_v4_prueba_bal": receta_original(args.dataset).get("bal_prueba"),
        "historial": historial,
        "tabla_fase1": tabla_fase1,
    }
    ruta = CARPETA_RESULTADOS / f"ajuste_{args.dataset}.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULTADO", json.dumps({"config": config_final.a_dict(), "epocas": epocas,
                                  "cv": mejor["cv_promedio"], "cv_semillas": mejor["cv_por_semilla"],
                                  "val": res_val, "prueba": res_test}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
