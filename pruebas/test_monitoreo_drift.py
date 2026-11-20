"""Tests for pipeline/monitoreo_drift.py. Run: python -m unittest pruebas.test_monitoreo_drift -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import monitoreo_drift as M  # noqa: E402


def datos(rng, n, desplazamiento=0.0):
    X = rng.normal(0, 1, (n, 3))
    X[:, 0] += desplazamiento
    pred = rng.choice(3, size=n, p=[0.85, 0.07, 0.08])
    ood = rng.random(n) < 0.01
    amb = rng.random(n) < 0.12
    return X, pred, ood, amb


class PsiTest(unittest.TestCase):
    def test_distribuciones_iguales_dan_cero(self):
        p = np.array([0.2, 0.3, 0.5])
        self.assertAlmostEqual(M.psi(p, p), 0.0, places=9)

    def test_crece_con_la_diferencia(self):
        ref = np.array([0.5, 0.5])
        self.assertLess(M.psi(ref, np.array([0.55, 0.45])), M.psi(ref, np.array([0.8, 0.2])))

    def test_proporciones_suman_uno_y_columna_binaria_tiene_un_borde(self):
        x = np.array([0.0] * 70 + [1.0] * 30)
        b = M.bordes(x)
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(M.proporciones(x, b).sum(), 1.0)


class MonitorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        Xtr, _, _, _ = datos(rng, 5000)
        self.Xv, self.pv, self.ov, self.av = datos(rng, 3000)
        self.ref = M.Referencia(Xtr, [0, 1, 2], self.pv, self.ov, self.av, K=3)

    def test_alarma_falsa_baja_en_ventanas_limpias(self):
        rng = np.random.default_rng(1)
        u = M.calibrar_umbrales(self.ref, self.Xv, self.pv, self.ov, self.av, n=300, n_ventanas=400, cuantil=0.995, rng=rng)
        Xt, pt, ot, at = datos(np.random.default_rng(2), 4000)
        alarmas = [M.alerta(self.ref.estadisticos(*M.ventana(rng, 300, Xt, pt, ot, at)), u)["alerta"] for _ in range(300)]
        self.assertLess(np.mean(alarmas), 0.10)

    def test_detecta_un_desplazamiento_de_dos_sd(self):
        rng = np.random.default_rng(3)
        u = M.calibrar_umbrales(self.ref, self.Xv, self.pv, self.ov, self.av, n=300, n_ventanas=400, cuantil=0.995, rng=rng)
        Xt, pt, ot, at = datos(np.random.default_rng(4), 4000, desplazamiento=2.0)
        r = M.alerta(self.ref.estadisticos(*M.ventana(rng, 300, Xt, pt, ot, at)), u)
        self.assertTrue(r["alerta"])
        self.assertIn("psi_max", r["disparadas"])

    def test_detecta_cambio_en_las_proporciones_de_clase(self):
        rng = np.random.default_rng(5)
        u = M.calibrar_umbrales(self.ref, self.Xv, self.pv, self.ov, self.av, n=300, n_ventanas=400, cuantil=0.995, rng=rng)
        X, _, o, a = datos(np.random.default_rng(6), 4000)
        pred = np.random.default_rng(7).choice(3, size=4000, p=[0.5, 0.25, 0.25])
        r = M.alerta(self.ref.estadisticos(*M.ventana(rng, 300, X, pred, o, a)), u)
        self.assertIn("psi_pred", r["disparadas"])

    def test_umbral_crece_cuando_la_ventana_se_achica(self):
        rng = np.random.default_rng(8)
        chica = M.calibrar_umbrales(self.ref, self.Xv, self.pv, self.ov, self.av, 100, 400, 0.995, rng)
        grande = M.calibrar_umbrales(self.ref, self.Xv, self.pv, self.ov, self.av, 1000, 400, 0.995, rng)
        self.assertGreater(chica["psi_max"], grande["psi_max"])


if __name__ == "__main__":
    unittest.main()
