"""Experimento: reentrenamiento de ds7 con la garantía capturada, sin el atajo de Valor_Entregado.

Entradas del formulario: campos de ds7 sin Valor_Entregado + Valor_Garantiza + Cobertura, donde
Cobertura = Valor_Garantiza / Valor_Entregado (la razón que define la clase, calculada en el formulario).
Excluidos: Tiene_Garantia (casi revela la clase) y Valor_Avaluo (no es un campo del formulario).

Particiones: las del dataset (columna Particion). El error de prueba no se usa para elegir nada.
Criterio de aceptación: brecha = error de prueba - error de entrenamiento < 3 pp (referencia: 5 pp).

Uso:  python pipeline/reentrenar_ds7_garantia.py
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
NUMERICAS = ["Anio_Desembolso", "Numero_Cuotas", "Tasa_Interes", "Cargas_Familiares", "Valor_Garantiza", "Cobertura"]
BINARIAS = ["Para_Vivienda", "Es_Agricola_Gan", "Es_Persona_Juridica", "Es_Dependiente"]
CATEGORICAS = ["Edad_Tramo", "Sexo", "Estado_Civil", "Nivel_Educacion", "Tipo_Vivienda", "Ingreso_Mensual_Tramo",
               "Total_Ingresos_Tramo", "Patrimonio_Tramo", "Categoria_Credito", "Nombre_Producto", "Sucursal", "Canton"]
SEMILLA = 42


def construir(df):
    X = pd.DataFrame(index=df.index)
    for c in NUMERICAS:
        if c == "Cobertura":
            X[c] = df["Valor_Garantiza"] / df["Valor_Entregado"]
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BINARIAS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X = pd.concat([X, pd.get_dummies(df[CATEGORICAS].astype(str), prefix=CATEGORICAS, dtype=float)], axis=1)
    return X.fillna(0.0)


def main():
    raw = pd.read_csv(I.V2_SALIDA / DS / "dataset_limpio.csv", low_memory=False)
    print("particiones:", raw["Particion"].value_counts().to_dict())
    X = construir(raw)
    y = raw["Y"].astype(int).to_numpy()
    part = raw["Particion"].astype(str).str.lower().to_numpy()

    tr = np.isin(part, ["train", "entrenamiento"])
    va = np.isin(part, ["val", "validacion", "validación"])
    te = np.isin(part, ["test", "prueba"])
    assert tr.sum() and va.sum() and te.sum(), "particiones no reconocidas"

    cols_num = [c for c in NUMERICAS]
    sc = StandardScaler().fit(X.loc[tr, cols_num])
    Xs = X.copy()
    Xs[cols_num] = sc.transform(X[cols_num])
    A = Xs.to_numpy(float)

    modelo = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, early_stopping=True,
                           validation_fraction=0.15, random_state=SEMILLA)
    modelo.fit(A[tr], y[tr])

    def err(m):
        p = modelo.predict(A[m])
        return 1 - float(np.mean(p == y[m])), float(balanced_accuracy_score(y[m], p))

    e_tr, b_tr = err(tr)
    e_va, b_va = err(va)
    e_te, b_te = err(te)
    brecha = e_te - e_tr
    print(f"error train {e_tr*100:.2f} %  val {e_va*100:.2f} %  prueba {e_te*100:.2f} %")
    print(f"exactitud balanceada train {b_tr:.3f}  val {b_va:.3f}  prueba {b_te:.3f}")
    print(f"BRECHA prueba - train = {brecha*100:+.2f} pp  ->  {'CUMPLE' if abs(brecha) < 0.03 else 'NO CUMPLE'} (< 3 pp); "
          f"{'cumple' if abs(brecha) < 0.05 else 'no cumple'} (< 5 pp)")
    print(f"columnas de entrada: {A.shape[1]}")


if __name__ == "__main__":
    main()
