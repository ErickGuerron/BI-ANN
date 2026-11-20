"""Tests for pipeline/filtro_ood.py and the rechazo_previo hook of politica_abstencion.
Run: python -m unittest pruebas.test_filtro_ood -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import filtro_ood as F  # noqa: E402
import politica_abstencion as PA  # noqa: E402


class KNNTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train = rng.normal(0, 1, (500, 4))
        self.filtro = F.FiltroKNN(k=5).ajustar(self.train)

    def test_lejano_puntua_mas_que_cercano(self):
        cerca = self.filtro.puntaje(np.zeros((1, 4)))[0]
        lejos = self.filtro.puntaje(np.full((1, 4), 20.0))[0]
        self.assertGreater(lejos, 10 * cerca)

    def test_percentil_rechaza_cerca_del_uno_por_ciento_de_validos(self):
        rng = np.random.default_rng(1)
        val = rng.normal(0, 1, (2000, 4))
        d_max = F.umbral_distancia(self.filtro.puntaje(val))
        test = rng.normal(0, 1, (2000, 4))
        tasa = F.rechaza(self.filtro.puntaje(test), d_max).mean()
        self.assertLess(tasa, 0.03)

    def test_rechaza_fuera_de_rango(self):
        d_max = F.umbral_distancia(self.filtro.puntaje(np.random.default_rng(2).normal(0, 1, (1000, 4))))
        fuera = np.full((10, 4), 15.0)
        self.assertTrue(F.rechaza(self.filtro.puntaje(fuera), d_max).all())

    def test_funciona_con_columnas_binarias(self):
        rng = np.random.default_rng(3)
        X = rng.integers(0, 2, (300, 6)).astype(float)
        f = F.FiltroKNN(k=3).ajustar(X)
        self.assertTrue(np.isfinite(f.puntaje(X[:5])).all())


class MahalanobisTest(unittest.TestCase):
    def test_covarianza_singular_no_rompe(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, (400, 3))
        Z = np.hstack([base, base[:, :1]])  # last column duplicates the first: rank 3 of 4
        y = rng.integers(0, 2, 400)
        f = F.FiltroMahalanobis().ajustar(Z, y, K=2)
        self.assertTrue(np.isfinite(f.puntaje(Z[:10])).all())

    def test_lejano_puntua_mas(self):
        rng = np.random.default_rng(1)
        Z = rng.normal(0, 1, (400, 3))
        y = rng.integers(0, 2, 400)
        f = F.FiltroMahalanobis().ajustar(Z, y, K=2)
        self.assertGreater(f.puntaje(np.full((1, 3), 12.0))[0], 10 * f.puntaje(np.zeros((1, 3)))[0])


class CascadaTest(unittest.TestCase):
    def test_rechazo_previo_fuerza_abstencion(self):
        P = np.array([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]])
        abst = PA.abstencion(P, umbral_h=0.9, rechazo_previo=np.array([False, True]))
        np.testing.assert_array_equal(abst, [False, True])

    def test_sin_rechazo_previo_no_cambia_el_comportamiento(self):
        P = np.array([[0.98, 0.01, 0.01], [0.34, 0.33, 0.33]])
        np.testing.assert_array_equal(PA.abstencion(P, 0.33), PA.abstencion(P, 0.33, rechazo_previo=None))


if __name__ == "__main__":
    unittest.main()
