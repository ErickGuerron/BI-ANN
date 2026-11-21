"""T6: two-step cascade for ds7. Step 1 geometric OOD filter, step 2 ensemble entropy with a per-class floor.

Detectors compared: k-NN on inputs (primary), class-conditional Mahalanobis with Ledoit-Wolf on the
penultimate layer, and the same on the inputs. d_max = 99th percentile on validation. The entropy threshold is
then chosen on validation, with the OOD rejections counted as not covered, under a 0.5 floor per class.
Test is scored once. Drift scenarios touch continuous columns only (binary columns cannot shift in practice),
plus a binary-flip scenario and extreme synthetic inputs.

Writes resultados_keras/t6_resumen.json. Requires the members saved by t5_abstencion_por_clase.py.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import keras  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_keras as T4  # noqa: E402
import politica_abstencion as PA  # noqa: E402
import filtro_ood as F  # noqa: E402
from t5_abstencion_por_clase import CARPETA_MIEMBROS, SALIDA, ensamble  # noqa: E402

PISO = 0.5
K = 3
K_VECINOS = 5
SEMILLA = 11


def cargar_miembros():
    rutas = [CARPETA_MIEMBROS / f"miembro_{s}.keras" for s in T4.MIEMBROS_ENSAMBLE]
    faltan = [r.name for r in rutas if not r.exists()]
    if faltan:
        raise FileNotFoundError(f"run t5_abstencion_por_clase.py first, missing: {faltan}")
    return [keras.models.load_model(r) for r in rutas]


def escenarios(Xt, cont, bin2, sigmas, valores_bin, rng):
    n, d = Xt.shape
    out = {}
    for s in (0.5, 1.0):
        X = Xt.copy()
        X[:, cont] += rng.normal(0, s, (n, len(cont))).astype(np.float32) * sigmas[cont]
        out[f"ruido_continuas_{s}sd"] = X
    for s in (1.0, 2.0, 3.0):
        X = Xt.copy()
        X[:, cont] += s * sigmas[cont]
        out[f"desplazamiento_continuas_{s}sd"] = X
    X = Xt.copy()
    for j in bin2:
        voltear = rng.random(n) < 0.10
        lo, hi = valores_bin[j]
        X[voltear, j] = np.where(np.isclose(X[voltear, j], lo), hi, lo)
    out["volteo_binarias_10pct"] = X
    X = Xt.copy()
    X[:, cont] *= 2.0
    out["escala_continuas_x2"] = X
    out["sintetico_uniforme_-4_4"] = rng.uniform(-4, 4, (n, d)).astype(np.float32)
    out["sintetico_normal_x5"] = rng.normal(0, 5, (n, d)).astype(np.float32)
    out["sintetico_ceros"] = np.zeros((n, d), dtype=np.float32)
    return out


def main():
    X, y = T4.cargar_datos(T4.DS7)
    modelos = cargar_miembros()

    nun = np.array([len(np.unique(X["train"][:, j])) for j in range(X["train"].shape[1])])
    cont = np.flatnonzero(nun > 3)
    bin2 = np.flatnonzero(nun == 2)
    valores_bin = {j: np.unique(X["train"][:, j]) for j in bin2}
    sigmas = X["train"].std(0).astype(np.float32)
    print("columnas continuas", len(cont), "binarias de 2 valores", len(bin2), flush=True)

    knn = F.FiltroKNN(K_VECINOS).ajustar(X["train"])
    Ztr = F.penultima(modelos, X["train"])
    mah_pen = F.FiltroMahalanobis().ajustar(Ztr, y["train"], K)
    mah_ent = F.FiltroMahalanobis().ajustar(X["train"], y["train"], K)
    detectores = {
        "knn_entradas": lambda Xs: knn.puntaje(Xs),
        "mahalanobis_penultima": lambda Xs: mah_pen.puntaje(F.penultima(modelos, Xs)),
        "mahalanobis_entradas_LW": lambda Xs: mah_ent.puntaje(Xs),
    }
    d_max = {n: F.umbral_distancia(f(X["val"])) for n, f in detectores.items()}
    print("d_max", d_max, flush=True)

    rng = np.random.default_rng(SEMILLA)
    esc = escenarios(X["test"], cont, bin2, sigmas, valores_bin, rng)
    esc_ev = {"prueba_original": X["test"], **esc}

    rechazo = {}
    for n, f in detectores.items():
        rechazo[n] = {nombre: round(float(F.rechaza(f(Xs), d_max[n]).mean()), 4) for nombre, Xs in esc_ev.items()}
        rechazo[n]["validacion"] = round(float(F.rechaza(f(X["val"]), d_max[n]).mean()), 4)
        print(n, rechazo[n], flush=True)

    # Cascade with the primary detector (k-NN).
    Pv = ensamble(modelos, X["val"])
    ood_val = F.rechaza(knn.puntaje(X["val"]), d_max["knn_entradas"])
    u_con, m_val_con = PA.elegir_umbral(Pv, y["val"], K, PISO, T4.GRID_UMBRAL, rechazo_previo=ood_val)
    u_sin, m_val_sin = PA.elegir_umbral(Pv, y["val"], K, PISO, T4.GRID_UMBRAL)
    cascada = {"umbral_H_con_ood": u_con, "umbral_H_sin_ood": u_sin, "val_con_ood": m_val_con, "val_sin_ood": m_val_sin}

    Pt = ensamble(modelos, X["test"])
    ood_test = F.rechaza(knn.puntaje(X["test"]), d_max["knn_entradas"])
    if u_con is not None:
        cascada["prueba_con_ood"] = PA.metricas(y["test"], Pt, PA.abstencion(Pt, u_con, rechazo_previo=ood_test), K)
    if u_sin is not None:
        cascada["prueba_sin_ood"] = PA.metricas(y["test"], Pt, PA.abstencion(Pt, u_sin), K)

    # Labeled drift scenarios: accepted-case quality with and without the OOD step.
    etiquetados = {}
    for nombre, Xs in esc.items():
        if nombre.startswith("sintetico"):
            continue
        P = ensamble(modelos, Xs)
        ood = F.rechaza(knn.puntaje(Xs), d_max["knn_entradas"])
        fila = {"rechazo_ood": round(float(ood.mean()), 4)}
        if u_con is not None:
            fila["con_ood"] = PA.metricas(y["test"], P, PA.abstencion(P, u_con, rechazo_previo=ood), K)
        if u_sin is not None:
            fila["sin_ood"] = PA.metricas(y["test"], P, PA.abstencion(P, u_sin), K)
        etiquetados[nombre] = fila

    resumen = {"piso_cobertura_por_clase": PISO, "k_vecinos": K_VECINOS, "percentil_d_max": F.PERCENTIL,
               "columnas_continuas": int(len(cont)), "d_max": d_max, "rechazo_por_detector": rechazo,
               "cascada": cascada, "escenarios_etiquetados": etiquetados}
    ruta = SALIDA / "t6_resumen.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
