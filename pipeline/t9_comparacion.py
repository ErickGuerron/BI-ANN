"""T9: same metrics, same scenarios, two ds7 ensembles: without and with Valor_Garantiza.

For each variant: ensemble quality on test, per-case cascade (k-NN filter + entropy, 0.5 floor per class),
and the population monitor recalibrated on that variant's own validation outputs. Drift scenarios are built
from the 96 original columns with the same seed, so both variants see identical perturbations; the
Valor_Garantiza column is left untouched. Thresholds always come from validation; test is scored once.

Writes resultados_keras/t9_comparacion.json. Run in the conda env `bi`.
"""
import json
import sys
from pathlib import Path

import keras
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import filtro_ood as F  # noqa: E402
import garantia as G  # noqa: E402
import monitoreo_drift as M  # noqa: E402
import politica_abstencion as PA  # noqa: E402
import t4_keras as T4  # noqa: E402
import t6_filtro_ood as T6  # noqa: E402
import t7_monitoreo_drift as T7  # noqa: E402
import t8_valor_garantiza as T8  # noqa: E402
from t5_abstencion_por_clase import SALIDA, ensamble  # noqa: E402

DS = T4.DS7
K = 3
TAMANOS = [300, 1000]
N_VENTANAS = 400
SEMILLA = 11


def recall(y, P):
    pred = P.argmax(1)
    return [round(float((pred[y == k] == k).mean()), 4) for k in range(K)]


def variante(nombre, X0, y, g):
    if nombre == "sin_garantia":
        modelos, X = T6.cargar_miembros(), {p: X0[p] for p in X0}
    else:
        modelos = [keras.models.load_model(T8.CARPETA_MIEMBROS / f"miembro_{s}.keras") for s in T4.MIEMBROS_ENSAMBLE]
        X = {p: G.agregar(DS, X0[p], g[p]).astype(np.float32) for p in X0}
    extra = [] if nombre == "sin_garantia" else [X["train"].shape[1] - 1]

    knn = F.FiltroKNN(T6.K_VECINOS).ajustar(X["train"])
    d_max = F.umbral_distancia(knn.puntaje(X["val"]))
    Pv, Pt = ensamble(modelos, X["val"]), ensamble(modelos, X["test"])
    ood_val = F.rechaza(knn.puntaje(X["val"]), d_max)
    ood_test = F.rechaza(knn.puntaje(X["test"]), d_max)
    u_h, _ = PA.elegir_umbral(Pv, y["val"], K, T6.PISO, T4.GRID_UMBRAL, rechazo_previo=ood_val)

    res = {"entradas": int(X["train"].shape[1]), "umbral_H": u_h, "d_max": d_max,
           "calidad_prueba": {**T4.metricas(y["test"], Pt), "recall": recall(y["test"], Pt), "ece": T4.ece(Pt, y["test"])},
           "cascada_prueba": PA.metricas(y["test"], Pt, PA.abstencion(Pt, u_h, rechazo_previo=ood_test), K)}

    nun = np.array([len(np.unique(X0["train"][:, j])) for j in range(X0["train"].shape[1])])
    cont0, bin2 = np.flatnonzero(nun > 3), np.flatnonzero(nun == 2)
    valores_bin = {j: np.unique(X0["train"][:, j]) for j in bin2}
    sigmas = X0["train"].std(0).astype(np.float32)
    rng = np.random.default_rng(SEMILLA)
    esc = T6.escenarios(X0["test"], cont0, bin2, sigmas, valores_bin, rng)
    escenarios = {n: (Xs, y["test"], g["test"]) for n, Xs in esc.items()}
    for factor in (0.5, 2.0):
        pesos = np.where(y["test"] == 0, 1.0, factor)
        idx = rng.choice(len(y["test"]), size=len(y["test"]), replace=True, p=pesos / pesos.sum())
        escenarios[f"cambio_prior_minorias_x{factor}"] = (X0["test"][idx], y["test"][idx], g["test"][idx])

    if nombre == "con_garantia":
        # Drift on the input that weighs most. Labels would change with the amount (the label is a rule on it),
        # so these scenarios report coverage and predicted classes only.
        for etiqueta, factor in (("garantia_x10", 10.0), ("garantia_x0.1", 0.1), ("garantia_en_blanco", 0.0)):
            escenarios[etiqueta] = (X0["test"], None, g["test"] * factor)

    def a_entrada(Xs, gs):
        return Xs if nombre == "sin_garantia" else G.agregar(DS, Xs, gs).astype(np.float32)

    val = T7.salidas(modelos, knn, d_max, u_h, X["val"])
    prueba = T7.salidas(modelos, knn, d_max, u_h, X["test"])
    ref = M.Referencia(X["train"], list(cont0) + extra, val["pred"], val["ood"], val["amb"], K)
    drift, por_caso = {}, {}
    for n_esc, (Xs, ys, gs) in escenarios.items():
        Xe = a_entrada(Xs, gs)
        drift[n_esc] = T7.salidas(modelos, knn, d_max, u_h, Xe)
        P = ensamble(modelos, Xe)
        abst = PA.abstencion(P, u_h, rechazo_previo=F.rechaza(knn.puntaje(Xe), d_max))
        if ys is not None and not n_esc.startswith("sintetico"):
            por_caso[n_esc] = PA.metricas(ys, P, abst, K)
        else:
            por_caso[n_esc] = {"cobertura": round(float((~abst).mean()), 4),
                               "prediccion": np.bincount(P.argmax(1), minlength=K).tolist()}

    monitor = {}
    for n in TAMANOS:
        u = M.calibrar_umbrales(ref, val["X"], val["pred"], val["ood"], val["amb"], n, N_VENTANAS, T7.CUANTIL, rng)
        falsas = np.mean([M.alerta(ref.estadisticos(*M.ventana(rng, n, prueba["X"], prueba["pred"], prueba["ood"],
                                                                prueba["amb"])), u)["alerta"] for _ in range(N_VENTANAS)])
        det = {}
        for n_esc, d in drift.items():
            det[n_esc] = {}
            for etiqueta, frac in (("completo", 1.0), ("contaminado_30", T7.CONTAMINACION)):
                r = [M.alerta(ref.estadisticos(*T7.mezclar(rng, n, prueba, d, frac)), u)["alerta"] for _ in range(N_VENTANAS)]
                det[n_esc][etiqueta] = round(float(np.mean(r)), 3)
        monitor[str(n)] = {"falsa_alarma": round(float(falsas), 4), "deteccion": det}
        print(nombre, "n", n, "falsa alarma", monitor[str(n)]["falsa_alarma"], flush=True)

    res["por_caso_bajo_drift"] = por_caso
    res["monitor"] = monitor
    return res


def main():
    X0, y = T4.cargar_datos(DS)
    g = T8.montos()
    salida = {nombre: variante(nombre, X0, y, g) for nombre in ("sin_garantia", "con_garantia")}
    ruta = SALIDA / "t9_comparacion.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
