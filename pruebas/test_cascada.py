"""Tests for nucleo/cascada.py with a tiny synthetic ensemble (no files, no TensorFlow).
Run: python -m unittest pruebas.test_cascada -v"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import cascada as C  # noqa: E402


def red_sintetica():
    """Two inputs, identity hidden layer, three classes. x=(2,0) is class 0 with certainty; x=(0,0) is uniform."""
    W0, b0 = np.eye(2), np.zeros(2)
    W1 = np.array([[4.0, -4.0, 0.0], [-4.0, 4.0, 0.0]])
    b1 = np.zeros(3)
    rng = np.random.default_rng(0)
    train = rng.uniform(0, 2, (300, 2)).astype(np.float32)
    cfg = {"miembros": 1, "k_vecinos": 3, "d_max": 1.0, "umbral_h": 0.5, "umbral_p": 0.65}
    return {"cfg": cfg, "miembros": [(W0, b0, W1, b1)], "nn": NearestNeighbors(n_neighbors=3).fit(train)}


class CascadaTest(unittest.TestCase):
    def evaluar(self, X):
        with mock.patch.object(C, "cargar", return_value=red_sintetica()):
            return C.evaluar("ds_falso", X)

    def test_confiado_y_dentro_de_rango_es_automatico(self):
        r = self.evaluar([[2.0, 0.0]])[0]
        self.assertEqual(r["estado"], C.AUTOMATICO)
        self.assertIsNone(r["motivo"])
        self.assertEqual(r["clase_idx"], 0)

    def test_lejos_del_entrenamiento_es_fuera_de_rango(self):
        r = self.evaluar([[50.0, 50.0]])[0]
        self.assertEqual(r["estado"], C.REVISION_MANUAL)
        self.assertEqual(r["motivo"], C.FUERA_DE_RANGO)

    def test_dentro_de_rango_pero_ambiguo_es_ambiguedad(self):
        r = self.evaluar([[0.0, 0.0]])[0]
        self.assertEqual(r["estado"], C.REVISION_MANUAL)
        self.assertEqual(r["motivo"], C.AMBIGUEDAD)
        self.assertAlmostEqual(r["entropia"], 1.0, places=6)

    def test_fuera_de_rango_tiene_prioridad_sobre_la_confianza(self):
        # Far along the class-0 direction the softmax is certain, but the geometric filter must still reject it.
        r = self.evaluar([[40.0, 0.0]])[0]
        self.assertGreater(max(r["probs"]), 0.99)
        self.assertEqual(r["motivo"], C.FUERA_DE_RANGO)

    def test_devuelve_una_fila_por_entrada_y_probabilidades_suman_uno(self):
        res = self.evaluar([[2.0, 0.0], [0.0, 0.0], [50.0, 50.0]])
        self.assertEqual(len(res), 3)
        for r in res:
            self.assertAlmostEqual(sum(r["probs"]), 1.0, places=6)

    def test_disponible_es_falso_sin_archivos(self):
        self.assertFalse(C.disponible("ds_que_no_existe"))


if __name__ == "__main__":
    unittest.main()
