"""T11: guarantee-specific drift statistics for the ds7 monitor (option 1 of the follow-up list).

Same ensemble with Valor_Garantiza as input (t8 members). Two monitors are compared on the SAME windows:
  old  psi_max, psi_pred, tasa_ood, desv_amb
  new  the old four + tasa_garantia_desv (share of rows carrying a guarantee) + ks_cobertura (KS on
       log(Valor_Garantiza / Valor_Entregado) among those rows)
Each threshold is the 0.995 quantile of validation windows of the same size (empirical null). False alarms are
measured on clean TEST windows; detection on windows built from drift scenarios, fully drifted or with 30 %
contamination. Scenarios on the other columns reuse T6; the new ones scale the guarantee amount only.

Writes resultados_keras/t11_monitoreo_garantia.json. Run in the conda env `bi`.
"""
import json
import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd

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
TAMANOS = [100, 300, 1000]
N_VENTANAS = 400
SEMILLA = 31
ORDEN = ("X", "pred", "ood", "amb", "g", "e")


def entregas():
    raw = pd.read_csv(T8.RAW, dtype={"Num_Operacion": str})
    raw["Valor_Entregado"] = pd.to_numeric(raw["Valor_Entregado"], errors="coerce")
    raw = raw.set_index("Num_Operacion")
    return {p: raw.loc[pd.read_csv(T4.V4 / DS / f"ids_{p}.csv", dtype=str)["Num_Operacion"], "Valor_Entregado"].to_numpy()
            for p in ("train", "val", "test")}


def mezclar(rng, n, limpio, drift, fraccion):
    n_d = int(round(n * fraccion))
    partes = []
    if n - n_d:
        partes.append(M.ventana(rng, n - n_d, *[limpio[k] for k in ORDEN]))
    if n_d:
        partes.append(M.ventana(rng, n_d, *[drift[k] for k in ORDEN]))
    return tuple(np.concatenate([p[i] for p in partes]) for i in range(len(ORDEN)))


def main():
    X0, y = T4.cargar_datos(DS)
    g, e = T8.montos(), entregas()
    modelos = [keras.models.load_model(T8.CARPETA_MIEMBROS / f"miembro_{s}.keras") for s in T4.MIEMBROS_ENSAMBLE]
    X = {p: G.agregar(DS, X0[p], g[p]).astype(np.float32) for p in X0}

    knn = F.FiltroKNN(T6.K_VECINOS).ajustar(X["train"])
    d_max = F.umbral_distancia(knn.puntaje(X["val"]))
    Pv = ensamble(modelos, X["val"])
    u_h, _ = PA.elegir_umbral(Pv, y["val"], K, T6.PISO, T4.GRID_UMBRAL, rechazo_previo=F.rechaza(knn.puntaje(X["val"]), d_max))

    def con_extras(Xe, ge, ee):
        d = T7.salidas(modelos, knn, d_max, u_h, Xe)
        d["g"], d["e"] = ge, ee
        return d

    val, prueba = con_extras(X["val"], g["val"], e["val"]), con_extras(X["test"], g["test"], e["test"])
    nun = np.array([len(np.unique(X0["train"][:, j])) for j in range(X0["train"].shape[1])])
    cont0, bin2 = np.flatnonzero(nun > 3), np.flatnonzero(nun == 2)
    valores_bin = {j: np.unique(X0["train"][:, j]) for j in bin2}
    sigmas = X0["train"].std(0).astype(np.float32)
    ref = M.Referencia(X["train"], list(cont0) + [X["train"].shape[1] - 1], val["pred"], val["ood"], val["amb"], K,
                       g_train=g["train"], e_train=e["train"])

    rng = np.random.default_rng(SEMILLA)
    esc = T6.escenarios(X0["test"], cont0, bin2, sigmas, valores_bin, rng)
    escenarios = {n: (Xs, g["test"], e["test"]) for n, Xs in esc.items()}
    for factor in (0.5, 2.0):
        pesos = np.where(y["test"] == 0, 1.0, factor)
        idx = rng.choice(len(y["test"]), size=len(y["test"]), replace=True, p=pesos / pesos.sum())
        escenarios[f"cambio_prior_minorias_x{factor}"] = (X0["test"][idx], g["test"][idx], e["test"][idx])
    for nombre, factor in (("garantia_x10", 10.0), ("garantia_x2", 2.0), ("garantia_x0.5", 0.5),
                           ("garantia_x0.1", 0.1), ("garantia_en_blanco", 0.0)):
        escenarios[nombre] = (X0["test"], g["test"] * factor, e["test"])
    drift = {n: con_extras(G.agregar(DS, Xs, gs).astype(np.float32), gs, es) for n, (Xs, gs, es) in escenarios.items()}

    arr_val = [val[k] for k in ORDEN]
    resultado = {"claves_viejas": list(M.CLAVES), "claves_nuevas": list(ref.claves), "por_tamano": {}}
    for n in TAMANOS:
        u = M.calibrar_umbrales(ref, *arr_val[:4], n, N_VENTANAS, T7.CUANTIL, rng, g=val["g"], e=val["e"])
        u_viejo = {k: u[k] for k in M.CLAVES}

        def evaluar(ventanas):
            est = [ref.estadisticos(*w) for w in ventanas]
            return (float(np.mean([M.alerta(x, u_viejo)["alerta"] for x in est])),
                    float(np.mean([M.alerta(x, u)["alerta"] for x in est])),
                    {k: round(float(np.mean([x[k] > u[k] for x in est])), 3) for k in ref.claves})

        limpias = [M.ventana(rng, n, *[prueba[k] for k in ORDEN]) for _ in range(N_VENTANAS)]
        v, nw, por = evaluar(limpias)
        fila = {"umbrales": {k: round(x, 4) for k, x in u.items()},
                "falsa_alarma": {"viejo": round(v, 4), "nuevo": round(nw, 4)}, "deteccion": {}}
        for nombre, d in drift.items():
            fila["deteccion"][nombre] = {}
            for etiqueta, frac in (("completo", 1.0), ("contaminado_30", T7.CONTAMINACION)):
                v, nw, por = evaluar([mezclar(rng, n, prueba, d, frac) for _ in range(N_VENTANAS)])
                fila["deteccion"][nombre][etiqueta] = {"viejo": round(v, 3), "nuevo": round(nw, 3), "por_estadistico": por}
        resultado["por_tamano"][str(n)] = fila
        print("n", n, "falsa alarma", fila["falsa_alarma"], flush=True)

    ruta = SALIDA / "t11_monitoreo_garantia.json"
    ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
