"""T5: per-class coverage floor for the ds7 abstention policy, and an out-of-distribution simulation.

1. Members: the 5 Keras networks of the ds7 ensemble (same recipe as T4). Trained once and saved to
   resultados_keras/t5_miembros/, then reloaded so perturbed inputs can be scored with the same models.
2. Viability: for each per-class coverage floor, the threshold is chosen on validation only. Test is scored once.
3. Simulated real environment: scenarios built from the ds7 test set (noise, shift, scale, label-prior change)
   and fully synthetic inputs outside the data range. Thresholds come from validation, not from the scenarios.

Writes resultados_keras/t5_resumen.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import keras  # noqa: E402

PROGRAMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as T4  # noqa: E402
import politica_abstencion as PA  # noqa: E402

SALIDA = PROGRAMA / "resultados_keras"
CARPETA_MIEMBROS = SALIDA / "t5_miembros"
COBERTURAS_MINIMAS = [0.3, 0.5, 0.7]
SEMILLA_ESCENARIOS = 7


def obtener_miembros(X, y, cfg, K):
    """Train the 5 members or reload them. Returns a list of Keras models."""
    CARPETA_MIEMBROS.mkdir(parents=True, exist_ok=True)
    modelos = []
    for semilla in T4.MIEMBROS_ENSAMBLE:
        ruta = CARPETA_MIEMBROS / f"miembro_{semilla}.keras"
        if ruta.exists():
            modelos.append(keras.models.load_model(ruta))
            continue
        import ann_core
        if cfg["sobremuestreo"]:
            Xt, yt = ann_core.sobremuestrear(X["train"], y["train"], semilla)
        else:
            Xt, yt = X["train"], y["train"]
        m = T4.entrenar(Xt, yt, X["val"], y["val"], K, cfg["alpha"], "ce", semilla)
        m.save(ruta)
        modelos.append(m)
        print("entrenado", semilla, flush=True)
    return modelos


def ensamble(modelos, X):
    return np.mean([T4.probas(m, X) for m in modelos], axis=0)


def escenarios_con_etiqueta(Xt, yt, sigmas, rng):
    """Labeled scenarios built from the test set. Labels stay the true ones."""
    out = {"prueba_original": (Xt, yt)}
    for s in (0.5, 1.0):
        out[f"ruido_gauss_{s}sd"] = (Xt + rng.normal(0, s, Xt.shape).astype(np.float32) * sigmas, yt)
    columnas = rng.choice(Xt.shape[1], size=int(0.2 * Xt.shape[1]), replace=False)
    desplazado = Xt.copy()
    desplazado[:, columnas] += 2.0 * sigmas[columnas]
    out["desplazamiento_20pct_cols_2sd"] = (desplazado, yt)
    out["escala_x2"] = (Xt * 2.0, yt)
    # Label-prior shift: minority classes at half their test frequency.
    factor = np.where(np.arange(yt.max() + 1) == 0, 1.0, 0.5)
    pesos = factor[yt]
    pesos = pesos / pesos.sum()
    idx = rng.choice(len(yt), size=len(yt), replace=True, p=pesos)
    out["cambio_prior_minorias_x0.5"] = (Xt[idx], yt[idx])
    return out


def evaluar_etiquetado(modelos, Xs, ys, umbral, K):
    P = ensamble(modelos, Xs)
    abst = PA.abstencion(P, umbral)
    sin_politica = PA.metricas(ys, P, np.zeros(len(ys), dtype=bool), K)
    con_politica = PA.metricas(ys, P, abst, K)
    return {"sin_politica": sin_politica, "con_politica": con_politica}


def evaluar_sin_etiqueta(modelos, Xs, umbral):
    P = ensamble(modelos, Xs)
    H = PA.entropia_normalizada(P)
    abst = PA.abstencion(P, umbral)
    return {
        "cobertura_con_politica": round(float((~abst).mean()), 4),
        "entropia_media": round(float(H.mean()), 4),
        "max_prob_media": round(float(P.max(1).mean()), 4),
        "cobertura_sin_politica": 1.0,
    }


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    X, y = T4.cargar_datos(T4.DS7)
    _, cfg = T4.cargar_config(T4.DS7)
    K = 3
    modelos = obtener_miembros(X, y, cfg, K)

    Pv = ensamble(modelos, X["val"])
    Pt = ensamble(modelos, X["test"])

    viabilidad = []
    elegidos = {}
    for c in COBERTURAS_MINIMAS:
        u, mv = PA.elegir_umbral(Pv, y["val"], K, c, T4.GRID_UMBRAL)
        fila = {"cobertura_min_clase": c, "umbral_H": u, "val": mv}
        if u is not None:
            mt = PA.metricas(y["test"], Pt, PA.abstencion(Pt, u), K)
            fila["prueba"] = mt
            elegidos[c] = u
        viabilidad.append(fila)
        print("cobertura_min", c, "umbral", u, flush=True)

    sigmas = X["train"].std(0).astype(np.float32)
    rng = np.random.default_rng(SEMILLA_ESCENARIOS)
    escenarios = {}
    for c, u in elegidos.items():
        etiq = {}
        for nombre, (Xs, ys) in escenarios_con_etiqueta(X["test"], y["test"], sigmas, rng).items():
            etiq[nombre] = evaluar_etiquetado(modelos, Xs, ys, u, K)
        escenarios[str(c)] = {"umbral_H": u, "etiquetados": etiq}

    # Fully synthetic inputs outside the data range: no labels, so only acceptance is measurable.
    fuera = {}
    d = X["test"].shape[1]
    sinteticos = {
        "uniforme_-4_4": rng.uniform(-4, 4, (5000, d)).astype(np.float32),
        "normal_x5": rng.normal(0, 5, (5000, d)).astype(np.float32),
        "ceros": np.zeros((5000, d), dtype=np.float32),
    }
    for c, u in elegidos.items():
        fuera[str(c)] = {nombre: evaluar_sin_etiqueta(modelos, Xs, u) for nombre, Xs in sinteticos.items()}

    resumen = {
        "miembros": len(modelos),
        "alpha_app": cfg["alpha"],
        "viabilidad": viabilidad,
        "escenarios_etiquetados": escenarios,
        "sinteticos_fuera_del_rango": fuera,
    }
    ruta = SALIDA / "t5_resumen.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
