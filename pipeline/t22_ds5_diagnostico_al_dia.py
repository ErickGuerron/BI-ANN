"""T22: why does the deployed ds5 network send the synthetic AL_DIA scenarios to MORA_LEVE?

Read-only diagnosis, nothing is retrained or tuned. Uses the 25 AL_DIA scenario cases of the external file
(see t21; same preparation, case 019 excluded).
  1. Nearest neighbours: for each case, the labels of its k nearest training rows in the network input space
     (the 112 standardised inputs the network sees) and its distance to the nearest training row, compared with
     the same distance for real validation rows.
  2. One-at-a-time substitution: replace one raw field by the typical value of real AL_DIA training rows
     (median for numbers, mode for text) and count how many cases flip to AL_DIA. Then replace whole groups.
Usage: python pipeline/t22_ds5_diagnostico_al_dia.py <xlsx>. Writes resultados_keras/t22_diagnostico_al_dia.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
PROGRAMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROGRAMA / "nucleo"))
import inferencia as I  # noqa: E402
import casos_externos  # noqa: E402

DS = "ds5_pagos_multimes"
K = 15
GRUPOS = {
    "comportamiento_de_pagos": ["N_Movimientos", "N_Cuotas_Con_Pago", "Ultima_Cuota_Pagada", "Monto_Total_Cobrado",
                                "Monto_Promedio_Mov", "Cuotas_Ult_3M", "Cuotas_Ult_6M", "Monto_Ult_3M", "Monto_Ult_6M",
                                "Saldo_Ultimo_Pago", "Dias_Desde_Ultimo_Pago", "Meses_Con_Pagos", "Pct_Cuotas_Pagadas"],
    "credito": ["Valor_Entregado", "Numero_Cuotas", "Tasa_Interes", "Categoria_Credito", "Nombre_Producto", "Sucursal",
                "Canton"],
    "perfil_cliente": ["Es_Persona_Juridica", "Edad_Tramo", "Sexo", "Estado_Civil", "Nivel_Educacion", "Tipo_Vivienda",
                       "Cargas_Familiares", "Ingreso_Mensual_Tramo", "Total_Ingresos_Tramo", "Patrimonio_Tramo",
                       "Es_Dependiente"],
}


def main(ruta):
    todo, ids, esperado, _ = casos_externos.preparar(ruta)
    entrada = todo[esperado == "AL_DIA"].reset_index(drop=True)
    nombres = I.cargar(DS)["aux"]["clases"]
    al = nombres.index("AL_DIA")
    base_cls, _ = I.predecir(DS, entrada)
    base_al = np.array([c == "AL_DIA" for c in base_cls])
    print("AL_DIA predicho en el original:", int(base_al.sum()), "de", len(entrada), flush=True)

    # 1. nearest neighbours in the network input space
    v4 = PROGRAMA.parent / DS
    Xtr = pd.read_csv(v4 / "X_train_nn.csv").to_numpy(float)
    ytr = pd.read_csv(v4 / "y_train.csv")["Y"].to_numpy().astype(int)
    Xva = pd.read_csv(v4 / "X_val_nn.csv").to_numpy(float)
    nn = NearestNeighbors(n_neighbors=K).fit(Xtr)
    Xc = I.entrada_red(DS, entrada).to_numpy(float)
    dist, ind = nn.kneighbors(Xc)
    etq = ytr[ind]
    vecinos = {n: round(float((etq == i).mean()), 3) for i, n in enumerate(nombres)}
    d_va, _ = nn.kneighbors(Xva[np.random.default_rng(0).choice(len(Xva), 300, replace=False)], n_neighbors=1)
    d_casos = dist[:, 0]
    print("etiqueta de los 15 vecinos (promedio sobre los 25 casos):", vecinos, flush=True)
    print("distancia al vecino mas cercano: casos mediana", round(float(np.median(d_casos)), 2),
          "| validacion real mediana", round(float(np.median(d_va)), 2), "p95", round(float(np.percentile(d_va, 95)), 2),
          "| casos por encima del p95 de validacion:", int((d_casos > np.percentile(d_va, 95)).sum()), flush=True)

    # 2. substitution toward typical real AL_DIA values
    tr = pd.read_csv(PROGRAMA.parents[1] / "datasets_multiclase_v2/salida/ds5_pagos_multimes/dataset_limpio.csv")
    tipico = tr[tr["Etiqueta"] == "AL_DIA"]
    valores = {}
    for c in entrada.columns:
        if c not in tipico.columns:
            continue
        s = tipico[c]
        valores[c] = float(s.median()) if pd.api.types.is_numeric_dtype(s) else s.mode().iloc[0]

    campos = {c["nombre"]: c for c in casos_externos.app.cargar_campos(DS)}

    def cuantos(columnas):
        e = entrada.copy()
        for c in columnas:
            if c in valores and c in campos:
                # same channel as the user: typed value -> app.revisar -> canonical value
                err, _, v = casos_externos.app.revisar(campos[c], casos_externos._bruto(campos[c], valores[c]))
                assert err is None, (c, err)
                e[c] = v
        cls, _ = I.predecir(DS, e)
        return int(sum(x == "AL_DIA" for x in cls))

    uno = {c: cuantos([c]) for c in entrada.columns if c in valores}
    efectivos = {c: v for c, v in uno.items() if v > int(base_al.sum())}
    print("campos que, cambiados solos, suben los AL_DIA predichos:", efectivos, flush=True)
    grupos = {g: cuantos(cols) for g, cols in GRUPOS.items()}
    grupos["todos_los_campos"] = cuantos(list(entrada.columns))
    print("cambio por grupo ->", grupos, flush=True)

    out = {"n_casos_AL_DIA": len(entrada), "AL_DIA_predicho_original": int(base_al.sum()), "k": K,
           "etiqueta_vecinos": vecinos, "dist_casos_mediana": round(float(np.median(d_casos)), 3),
           "dist_validacion_mediana": round(float(np.median(d_va)), 3),
           "dist_validacion_p95": round(float(np.percentile(d_va, 95)), 3),
           "casos_sobre_p95": int((d_casos > np.percentile(d_va, 95)).sum()),
           "AL_DIA_tras_cambiar_un_campo": uno, "AL_DIA_tras_cambiar_un_grupo": grupos,
           "valores_tipicos_usados": {k: v for k, v in valores.items()}}
    (PROGRAMA / "resultados_keras" / "t22_diagnostico_al_dia.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
