"""Tests for pipeline/politica_abstencion.py. Run: python -m unittest pruebas.test_politica_abstencion -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import politica_abstencion as PA  # noqa: E402


class EntropiaTest(unittest.TestCase):
    def test_uniforme_da_uno(self):
        P = np.full((1, 3), 1 / 3)
        self.assertAlmostEqual(PA.entropia_normalizada(P)[0], 1.0, places=6)

    def test_one_hot_da_cero(self):
        P = np.array([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(PA.entropia_normalizada(P)[0], 0.0, places=6)


class AbstencionTest(unittest.TestCase):
    def test_confiado_se_acepta(self):
        P = np.array([[0.98, 0.01, 0.01]])
        self.assertFalse(PA.abstencion(P, umbral_h=0.33)[0])

    def test_entropia_alta_se_abstiene(self):
        P = np.array([[0.34, 0.33, 0.33]])
        self.assertTrue(PA.abstencion(P, umbral_h=0.33)[0])

    def test_maximo_bajo_se_abstiene_aunque_la_entropia_pase(self):
        # H_n about 0.66 < 0.99 so only the 0.65 max-probability rule can abstain here.
        P = np.array([[0.6, 0.39, 0.01]])
        self.assertTrue(PA.abstencion(P, umbral_h=0.99)[0])


class CoberturaTest(unittest.TestCase):
    def test_cobertura_por_clase(self):
        y = np.array([0, 0, 1, 1, 2])
        abst = np.array([False, True, False, False, True])
        cob = PA.cobertura_por_clase(y, abst, K=3)
        np.testing.assert_allclose(cob, [0.5, 1.0, 0.0])

    def test_clase_ausente_es_nan(self):
        cob = PA.cobertura_por_clase(np.array([0, 0]), np.array([False, False]), K=2)
        self.assertTrue(np.isnan(cob[1]))


class ElegirUmbralTest(unittest.TestCase):
    def setUp(self):
        # Class 0 and 1 are confident and correct; class 2 is always uncertain (uniform), so its coverage is 0.
        self.K = 3
        self.Pv = np.array([[0.98, 0.01, 0.01]] * 20 + [[0.01, 0.98, 0.01]] * 20 + [[1 / 3, 1 / 3, 1 / 3]] * 20)
        self.yv = np.array([0] * 20 + [1] * 20 + [2] * 20)
        self.grilla = np.round(np.arange(0.05, 1.0001, 0.05), 4)

    def test_infactible_si_una_clase_nunca_se_acepta(self):
        u, m = PA.elegir_umbral(self.Pv, self.yv, self.K, cobertura_min_clase=0.5, grilla=self.grilla)
        self.assertIsNone(u)
        self.assertIsNone(m)

    def test_factible_sin_restriccion(self):
        u, m = PA.elegir_umbral(self.Pv, self.yv, self.K, cobertura_min_clase=0.0, grilla=self.grilla)
        self.assertIsNotNone(u)
        self.assertGreaterEqual(m["cobertura"], 0.0)

    def test_respeta_el_minimo_por_clase(self):
        Pv = np.vstack([self.Pv[:20], self.Pv[20:40], np.array([[0.98, 0.01, 0.01]] * 20)])
        yv = np.array([0] * 20 + [1] * 20 + [2] * 20)
        u, m = PA.elegir_umbral(Pv, yv, self.K, cobertura_min_clase=0.9, grilla=self.grilla)
        self.assertIsNotNone(u)
        self.assertGreaterEqual(min(m["cobertura_por_clase"]), 0.9)


if __name__ == "__main__":
    unittest.main()
