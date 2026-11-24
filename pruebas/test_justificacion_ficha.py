"""Justification tab of the network sheet: Hoeffding margins, network size and extra measures.
Run: python -m unittest pruebas.test_justificacion_ficha -v"""
import math
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import ficha_red  # noqa: E402

DATASETS = ["ds1_ingreso_solicitudes", "ds2_ciclo_solicitud", "ds3_desembolsos", "ds4_cierre_cartera_mensual",
            "ds5_pagos_multimes", "ds6_gasto_provisiones", "ds7_historico_garantias", "ds8_notificaciones_cobranza"]


class HoeffdingTest(unittest.TestCase):
    def test_margen_formula(self):
        n = 2744
        esperado = math.sqrt(math.log(40) / (2 * n))
        self.assertAlmostEqual(ficha_red.margen_hoeffding(n), esperado, places=12)
        self.assertAlmostEqual(ficha_red.margen_hoeffding(n), 0.0259, places=3)

    def test_n_para_margen_conocido(self):
        # ln(40) / (2 · margen²): 4 611,1 para ±2 pp y 18 444,4 para ±1 pp; se redondea hacia arriba
        self.assertEqual(ficha_red.n_hoeffding(0.02), 4612)
        self.assertEqual(ficha_red.n_hoeffding(0.01), 18445)

    def test_n_suficiente_cumple_el_margen_pedido(self):
        for margen in (0.05, 0.02, 0.01):
            n = ficha_red.n_hoeffding(margen)
            self.assertLessEqual(ficha_red.margen_hoeffding(n), margen)
            self.assertGreater(ficha_red.margen_hoeffding(n - 1), margen)

    def test_margen_baja_con_mas_filas(self):
        self.assertGreater(ficha_red.margen_hoeffding(1868), ficha_red.margen_hoeffding(7271))


class DimensionTest(unittest.TestCase):
    def test_cuenta_pesos_y_sesgos_por_capa(self):
        pesos = {"entradas": ["a"] * 112, "capas": [112, 32, 4],
                 "coefs": [np.zeros((112, 32)), np.zeros((32, 4))],
                 "intercepts": [np.zeros(32), np.zeros(4)]}
        d = ficha_red.dimension_red(pesos, 21601)
        self.assertEqual(d["parametros_por_capa"], [112 * 32 + 32, 32 * 4 + 4])
        self.assertEqual(d["parametros"], 3748)
        self.assertEqual(d["regla_10_por_parametro"], 37480)
        self.assertAlmostEqual(d["muestras_por_parametro"], 21601 / 3748, places=9)

    def test_sin_datos_de_entrenamiento_no_divide(self):
        pesos = {"entradas": ["a"], "capas": [1, 1], "coefs": [np.zeros((1, 1))], "intercepts": [np.zeros(1)]}
        self.assertIsNone(ficha_red.dimension_red(pesos, None)["muestras_por_parametro"])


class MetricasExtraTest(unittest.TestCase):
    def test_mcc_coincide_con_sklearn_en_varias_clases(self):
        rng = np.random.default_rng(0)
        real = rng.integers(0, 4, 500)
        pred = np.where(rng.random(500) < 0.6, real, rng.integers(0, 4, 500))
        cm = np.zeros((4, 4), int)
        for a, b in zip(real, pred):
            cm[a, b] += 1
        self.assertAlmostEqual(ficha_red.mcc_multiclase(cm), matthews_corrcoef(real, pred), places=10)

    def test_mcc_perfecto_y_azar(self):
        self.assertAlmostEqual(ficha_red.mcc_multiclase([[5, 0], [0, 7]]), 1.0)
        # predecir siempre la clase mayoritaria: el denominador es 0 y el MCC vale 0
        self.assertEqual(ficha_red.mcc_multiclase([[0, 0], [3, 9]]), 0.0)

    def test_brier_perfecto_y_con_error(self):
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(ficha_red.brier_multiclase(P, np.array([0, 1])), 0.0)
        # confianza 0,5 en ambas filas: (0,5² + 0,5²) = 0,5 por fila
        P2 = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.assertAlmostEqual(ficha_red.brier_multiclase(P2, np.array([0, 1])), 0.5)

    def test_ece_cero_si_la_confianza_coincide_con_el_acierto(self):
        # confianza 1 y siempre acierta: no hay error de calibración
        P = np.array([[1.0, 0.0]] * 4)
        self.assertAlmostEqual(ficha_red.ece_top_label(P, np.array([0, 0, 0, 0])), 0.0)
        # confianza 1 y siempre falla: error de 1
        self.assertAlmostEqual(ficha_red.ece_top_label(P, np.array([1, 1, 1, 1])), 1.0)


class PreguntaCartaTest(unittest.TestCase):
    def test_cada_red_tiene_una_pregunta(self):
        for ds in DATASETS:
            self.assertIn(ds, ficha_red.PREGUNTA_CARTA)
            self.assertTrue(ficha_red.PREGUNTA_CARTA[ds].strip())

    def test_ficha_incluye_justificacion_para_ds5(self):
        j = ficha_red.cargar_ficha("ds5_pagos_multimes")["justificacion"]
        self.assertEqual(j["n_prueba"], 7271)
        self.assertEqual(j["dimension"]["parametros"], 3748)
        self.assertEqual(j["dimension"]["entradas"], 112)
        self.assertEqual(j["dimension"]["muestras_train"], 21601)
        self.assertAlmostEqual(j["hoeffding"]["margen_error"], 0.0159, places=3)
        self.assertGreater(j["extra"]["brier"], 0)
        self.assertGreaterEqual(j["extra"]["ece"], 0)
        self.assertIn("Pregunta 1", j["pregunta"])


class ValidacionCruzadaTest(unittest.TestCase):
    def test_k_es_el_de_la_busqueda(self):
        # pipeline/ajuste_convergencia.py: --folds por defecto 5, sin sobrescribirlo en ningún script
        self.assertEqual(ficha_red.CV_K, 5)

    def test_filas_por_pliegue_son_el_entrenamiento_dividido_entre_k(self):
        vc = ficha_red.cargar_ficha("ds5_pagos_multimes")["justificacion"]["validacion_cruzada"]
        self.assertEqual(vc["k"], 5)
        self.assertAlmostEqual(vc["filas_por_pliegue"], 21601 / 5, places=6)
        self.assertGreater(vc["margen_por_pliegue"], ficha_red.margen_hoeffding(7271))

    def test_indicador_de_margen_2pp(self):
        # ds5 y ds6 tienen prueba mayor que 4 612 filas; ds1 y ds7 no
        self.assertTrue(ficha_red.cargar_ficha("ds5_pagos_multimes")["justificacion"]["hoeffding"]["cumple_2_pp"])
        self.assertTrue(ficha_red.cargar_ficha("ds6_gasto_provisiones")["justificacion"]["hoeffding"]["cumple_2_pp"])
        self.assertFalse(ficha_red.cargar_ficha("ds7_historico_garantias")["justificacion"]["hoeffding"]["cumple_2_pp"])
        self.assertFalse(ficha_red.cargar_ficha("ds1_ingreso_solicitudes")["justificacion"]["hoeffding"]["cumple_2_pp"])


if __name__ == "__main__":
    unittest.main()
