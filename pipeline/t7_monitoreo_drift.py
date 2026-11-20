"""T7: viability of the population-level drift monitor on ds7.

Per-case outputs (predicted class, OOD flag, ambiguity flag) come from the ensemble and the k-NN filter of T6.
Thresholds are calibrated on validation windows (empirical null, quantile 0.995) for each window size.
Evaluation uses test only: false-alarm rate on clean windows, and detection rate on windows built from drift
scenarios, either fully drifted or contaminated (30 % drifted, 70 % clean).

Writes resultados_keras/t7_resumen.json. Requires the members saved by t5_abstencion_por_clase.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as T4  # noqa: E402
import politica_abstencion as PA  # noqa: E402
import filtro_ood as F  # noqa: E402
import monitoreo_drift as M  # noqa: E402
from t5_abstencion_por_clase import SALIDA, ensamble  # noqa: E402
from t6_filtro_ood import K, PISO, K_VECINOS, cargar_miembros, escenarios  # noqa: E402

TAMANOS = [100, 300, 1000]
N_VENTANAS = 400
CUANTIL = 0.995
CONTAMINACION = 0.30
SEMILLA = 21


def salidas(modelos, knn, d_max, u_h, X):
    P = ensamble(modelos, X)
    H = PA.entropia_normalizada(P)
    return {"X": X, "pred": P.argmax(1), "ood": F.rechaza(knn.puntaje(X), d_max),
            "amb": (H > u_h) | (P.max(1) < PA.UMBRAL_P)}


def mezclar(rng, n, limpio, drift, fraccion):
    """Window with `fraccion` of its cases drawn from the drifted arrays and the rest from the clean ones."""
    n_d = int(round(n * fraccion))
    a = M.ventana(rng, n - n_d, limpio["X"], limpio["pred"], limpio["ood"], limpio["amb"]) if n - n_d else None
    b = M.ventana(rng, n_d, drift["X"], drift["pred"], drift["ood"], drift["amb"]) if n_d else None
    partes = [p for p in (a, b) if p is not None]
    return tuple(np.concatenate([p[i] for p in partes]) for i in range(4))


def main():
    X, y = T4.cargar_datos(T4.DS7)
    modelos = cargar_miembros()
    knn = F.FiltroKNN(K_VECINOS).ajustar(X["train"])
    d_max = F.umbral_distancia(knn.puntaje(X["val"]))
    Pv = ensamble(modelos, X["val"])
    u_h, _ = PA.elegir_umbral(Pv, y["val"], K, PISO, T4.GRID_UMBRAL, rechazo_previo=F.rechaza(knn.puntaje(X["val"]), d_max))

    nun = np.array([len(np.unique(X["train"][:, j])) for j in range(X["train"].shape[1])])
    cont, bin2 = np.flatnonzero(nun > 3), np.flatnonzero(nun == 2)
    valores_bin = {j: np.unique(X["train"][:, j]) for j in bin2}
    sigmas = X["train"].std(0).astype(np.float32)

    val = salidas(modelos, knn, d_max, u_h, X["val"])
    prueba = salidas(modelos, knn, d_max, u_h, X["test"])
    ref = M.Referencia(X["train"], cont, val["pred"], val["ood"], val["amb"], K)

    rng = np.random.default_rng(SEMILLA)
    esc = escenarios(X["test"], cont, bin2, sigmas, valores_bin, rng)
    for factor in (0.5, 2.0):
        pesos = np.where(y["test"] == 0, 1.0, factor)
        idx = rng.choice(len(y["test"]), size=len(y["test"]), replace=True, p=pesos / pesos.sum())
        esc[f"cambio_prior_minorias_x{factor}"] = X["test"][idx]
    drift = {nombre: salidas(modelos, knn, d_max, u_h, Xs) for nombre, Xs in esc.items()}

    resultado = {"umbral_H": u_h, "d_max": d_max, "cuantil_umbral": CUANTIL, "contaminacion": CONTAMINACION,
                 "columnas_continuas": int(len(cont)), "por_tamano": {}}
    for n in TAMANOS:
        u = M.calibrar_umbrales(ref, val["X"], val["pred"], val["ood"], val["amb"], n, N_VENTANAS, CUANTIL, rng)
        falsas = [M.alerta(ref.estadisticos(*M.ventana(rng, n, prueba["X"], prueba["pred"], prueba["ood"], prueba["amb"])), u)
                  for _ in range(N_VENTANAS)]
        fila = {"umbrales": {k: round(v, 4) for k, v in u.items()},
                "alarma_falsa_ventanas_limpias": round(float(np.mean([f["alerta"] for f in falsas])), 4),
                "deteccion": {}}
        for nombre, d in drift.items():
            fila["deteccion"][nombre] = {}
            for etiqueta, frac in (("completo", 1.0), ("contaminado_30", CONTAMINACION)):
                res = [M.alerta(ref.estadisticos(*mezclar(rng, n, prueba, d, frac)), u) for _ in range(N_VENTANAS)]
                disp = {k: round(float(np.mean([k in r["disparadas"] for r in res])), 3) for k in M.CLAVES}
                fila["deteccion"][nombre][etiqueta] = {"deteccion": round(float(np.mean([r["alerta"] for r in res])), 4),
                                                       "por_estadistico": disp}
        resultado["por_tamano"][str(n)] = fila
        print("n", n, "umbrales", fila["umbrales"], "falsa alarma", fila["alarma_falsa_ventanas_limpias"], flush=True)

    ruta = SALIDA / "t7_resumen.json"
    ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
