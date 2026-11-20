"""ds7 con garantía capturada: modelo de ML puro, sin ninguna variable derivada de la regla.

Entradas: campos del formulario de ds7 (sin Valor_Entregado como atajo de cobertura) + Valor_Garantiza y
Valor_Entregado en bruto. NO se calcula la cobertura: la red debe aprender por sí misma la relación entre ambos.
Excluidos: Tiene_Garantia (revela la clase) y Valor_Avaluo (no es un campo del formulario).

Evaluación:
  1. Métricas en las particiones del dataset (train, val, prueba). Prueba no se usa para elegir nada.
  2. Sondeo de generalización sobre filas de PRUEBA con la garantía reemplazada por k * Valor_Entregado,
     para k en {0, 0.5, 1.5, 2.5, 4}. La regla solo se usa aquí para calificar la respuesta; nunca entra
     al modelo. Con k = 0 la etiqueta esperada es SIN_GARANTIA; si 0 < k <= 2, COBERTURA_BAJA; si k > 2, COBERTURA_ALTA.

Criterio de aceptación: brecha = error de prueba - error de entrenamiento < 3 pp (referencia 5 pp).

Uso: python pipeline/ds7_garantia_ml_puro.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import inferencia as I  # noqa: E402

DS = "ds7_historico_garantias"
NUMERICAS = ["Anio_Desembolso", "Numero_Cuotas", "Tasa_Interes", "Cargas_Familiares", "Valor_Garantiza", "Valor_Entregado"]
BINARIAS = ["Para_Vivienda", "Es_Agricola_Gan", "Es_Persona_Juridica", "Es_Dependiente"]
CATEGORICAS = ["Edad_Tramo", "Sexo", "Estado_Civil", "Nivel_Educacion", "Tipo_Vivienda", "Ingreso_Mensual_Tramo",
               "Total_Ingresos_Tramo", "Patrimonio_Tramo", "Categoria_Credito", "Nombre_Producto", "Sucursal", "Canton"]
SEMILLA = 42
K_SONDEO = [0.0, 0.5, 1.5, 2.5, 4.0]


def construir(df):
    X = pd.DataFrame(index=df.index)
    for c in NUMERICAS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BINARIAS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X = pd.concat([X, pd.get_dummies(df[CATEGORICAS].astype(str), prefix=CATEGORICAS, dtype=float)], axis=1)
    return X.fillna(0.0)


def etiqueta_esperada(k):
    """Solo para calificar el sondeo. El modelo no recibe esta información."""
    if k == 0:
        return "SIN_GARANTIA"
    return "COBERTURA_BAJA" if k <= 2 else "COBERTURA_ALTA"


def main():
    raw = pd.read_csv(I.V2_SALIDA / DS / "dataset_limpio.csv", low_memory=False)
    X = construir(raw)
    y = raw["Y"].astype(int).to_numpy()
    clases = I.cargar(DS)["aux"]["clases"]
    part = raw["Particion"].astype(str).str.lower().to_numpy()
    tr, va, te = part == "train", part == "val", part == "test"

    sc = StandardScaler().fit(X.loc[tr, NUMERICAS])
    Xs = X.copy()
    Xs[NUMERICAS] = sc.transform(X[NUMERICAS])
    modelo = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, early_stopping=True,
                           validation_fraction=0.15, random_state=SEMILLA)
    modelo.fit(Xs.to_numpy(float)[tr], y[tr])

    def err(m):
        p = modelo.predict(Xs.to_numpy(float)[m])
        return 1 - float(np.mean(p == y[m])), float(balanced_accuracy_score(y[m], p))

    e_tr, b_tr = err(tr)
    e_va, b_va = err(va)
    e_te, b_te = err(te)
    brecha = e_te - e_tr
    print(f"error train {e_tr*100:.2f} %  val {e_va*100:.2f} %  prueba {e_te*100:.2f} %")
    print(f"exactitud balanceada train {b_tr:.3f}  val {b_va:.3f}  prueba {b_te:.3f}")
    print(f"BRECHA prueba - train = {brecha*100:+.2f} pp -> {'CUMPLE' if abs(brecha) < 0.03 else 'NO CUMPLE'} (< 3 pp)")

    print("\nSondeo sobre prueba con garantía = k * Valor_Entregado (k no visto en entrenamiento):")
    print(f"{'k':>5} {'esperada':>16} {'acierto':>8} {'confianza media':>16} {'clase mas frecuente':>24}")
    idx = np.where(te)[0]
    for k in K_SONDEO:
        R = raw.iloc[idx].copy()
        R["Valor_Garantiza"] = (k * pd.to_numeric(R["Valor_Entregado"], errors="coerce")).round(0)
        Xk = construir(R).reindex(columns=X.columns, fill_value=0.0)
        Xk[NUMERICAS] = sc.transform(Xk[NUMERICAS])
        pr = modelo.predict_proba(Xk.to_numpy(float))
        pred = np.array([clases[i] for i in pr.argmax(1)])
        esp = etiqueta_esperada(k)
        acierto = float(np.mean(pred == esp))
        conf = float(pr.max(1).mean())
        mas = pd.Series(pred).value_counts(normalize=True).index[0]
        print(f"{k:5.1f} {esp:>16} {acierto*100:7.1f}% {conf*100:15.1f}% {mas:>24}")


if __name__ == "__main__":
    main()
