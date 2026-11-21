"""Money fields accept decimals; counts and years stay integer-only.
Run: python -m unittest pruebas.test_app_monetarios -v"""
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402

TOP4 = ["ds7_historico_garantias", "ds1_ingreso_solicitudes", "ds6_gasto_provisiones", "ds5_pagos_multimes"]
ENTEROS = {"Numero_Cuotas", "Anio", "Anio_Desembolso", "N_Movimientos", "Dias_Morosidad", "Cuotas_Ult_3M"}


class MonetariosTest(unittest.TestCase):
    def campos(self, ds):
        return {c["nombre"]: c for c in app.cargar_campos(ds) if c["tipo"] == "num"}

    def test_los_campos_monetarios_no_son_enteros_en_ningun_dataset(self):
        for ds in TOP4:
            for nombre, c in self.campos(ds).items():
                if app.I.es_monetario(nombre):
                    self.assertFalse(c["entero"], f"{ds}.{nombre}")

    def test_los_monetarios_conocidos_estan_cubiertos(self):
        esperados = {"ds1_ingreso_solicitudes": {"Total_Ingresos", "Total_Gastos", "Monto_Solicitado"},
                     "ds5_pagos_multimes": {"Valor_Entregado", "Monto_Total_Cobrado", "Monto_Ult_3M", "Monto_Ult_6M",
                                            "Saldo_Ultimo_Pago"},
                     "ds6_gasto_provisiones": {"Saldo_Actual", "Saldo_Vencido", "Valor_Entregado",
                                               "Provision_Constituida", "Provision_Automatica", "Provision_Manual"},
                     "ds7_historico_garantias": {"Valor_Entregado", "Valor_Garantiza"}}
        for ds, nombres in esperados.items():
            self.assertTrue(nombres <= set(self.campos(ds)), ds)
            self.assertTrue(all(not self.campos(ds)[n]["entero"] for n in nombres), ds)

    def test_un_monto_con_decimales_se_acepta_y_se_conserva(self):
        c = self.campos("ds7_historico_garantias")["Valor_Entregado"]
        err, _, valor = app.revisar(c, 1500.75)
        self.assertIsNone(err)
        self.assertEqual(valor, 1500.75)

    def test_conteos_y_anios_siguen_siendo_enteros(self):
        for ds in TOP4:
            for nombre, c in self.campos(ds).items():
                if nombre in ENTEROS:
                    self.assertTrue(c["entero"], f"{ds}.{nombre}")
                    err, _, _ = app.revisar(c, c["min"] + 0.5)
                    self.assertIn("entero", err)

    def test_el_monto_negativo_sigue_bloqueado_salvo_saldo_actual(self):
        c = self.campos("ds7_historico_garantias")["Valor_Entregado"]
        self.assertIsNotNone(app.revisar(c, -1.5)[0])
        self.assertTrue(app.I.es_monetario("Saldo_Actual"))


if __name__ == "__main__":
    unittest.main()
