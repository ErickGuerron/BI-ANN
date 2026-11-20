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


def datos_g(rng, n, factor=1.0, blanco=False):
    """Toy window with guarantee amounts: about 8 % of the rows have one, coverage around 2 on a log scale."""
    X, pred, ood, amb = datos(rng, n)
    e = rng.uniform(500, 20000, n)
    tiene = rng.random(n) < 0.08
    k = np.exp(rng.normal(np.log(2.0), 0.6, n))
    g = np.where(tiene, np.round(k * e), 0.0) * factor
    if blanco:
        g = np.zeros(n)
    return X, pred, ood, amb, g, e


class GarantiaTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        Xtr, _, _, _, gtr, etr = datos_g(rng, 8000)
        self.Xv, self.pv, self.ov, self.av, self.gv, self.ev = datos_g(rng, 4000)
        self.ref = M.Referencia(Xtr, [0, 1, 2], self.pv, self.ov, self.av, K=3, g_train=gtr, e_train=etr)

    def umbrales(self, n, rng):
        return M.calibrar_umbrales(self.ref, self.Xv, self.pv, self.ov, self.av, n, 400, 0.995, rng, g=self.gv, e=self.ev)

    def test_ks_identicas_cero_y_separadas_uno(self):
        a = np.arange(50.0)
        self.assertAlmostEqual(M.ks(a, a), 0.0)
        self.assertAlmostEqual(M.ks(a, a + 1000), 1.0)

    def test_los_estadisticos_incluyen_las_dos_claves_de_garantia(self):
        est = self.ref.estadisticos(*datos_g(np.random.default_rng(1), 300))
        self.assertIn("tasa_garantia_desv", est)
        self.assertIn("ks_cobertura", est)

    def test_sin_garantia_en_la_referencia_no_agrega_estadisticos(self):
        ref = M.Referencia(self.Xv, [0, 1, 2], self.pv, self.ov, self.av, K=3)
        est = ref.estadisticos(*datos(np.random.default_rng(1), 300))
        self.assertNotIn("ks_cobertura", est)

    def test_ventana_sin_garantias_positivas_da_ks_cero(self):
        X, p, o, a, g, e = datos_g(np.random.default_rng(2), 300, blanco=True)
        est = self.ref.estadisticos(X, p, o, a, g, e)
        self.assertEqual(est["ks_cobertura"], 0.0)
        p0 = self.ref.tasa_garantia_
        self.assertAlmostEqual(est["tasa_garantia_desv"], p0 / np.sqrt(p0 * (1 - p0) / 300), places=6)

    def test_los_umbrales_de_garantia_son_analiticos_y_no_dependen_de_la_calibracion(self):
        for n in (100, 1000):
            u = self.umbrales(n, np.random.default_rng(n))
            self.assertEqual(u["tasa_garantia_desv"], M.Z_UMBRAL)
            self.assertEqual(u["ks_cobertura"], M.KS_UMBRAL)

    def test_el_ks_escalado_crece_con_el_numero_de_positivas(self):
        # The same shift is more convincing with more positive rows in the window.
        chico = self.ref.estadisticos(*datos_g(np.random.default_rng(11), 200, factor=3.0))["ks_cobertura"]
        grande = self.ref.estadisticos(*datos_g(np.random.default_rng(11), 4000, factor=3.0))["ks_cobertura"]
        self.assertGreater(grande, chico)

    def test_detecta_la_garantia_en_blanco(self):
        rng = np.random.default_rng(3)
        u = self.umbrales(300, rng)
        r = M.alerta(self.ref.estadisticos(*datos_g(np.random.default_rng(4), 300, blanco=True)), u)
        self.assertIn("tasa_garantia_desv", r["disparadas"])

    def test_detecta_la_garantia_multiplicada_por_diez(self):
        rng = np.random.default_rng(5)
        u = self.umbrales(1000, rng)
        r = M.alerta(self.ref.estadisticos(*datos_g(np.random.default_rng(6), 1000, factor=10.0)), u)
        self.assertIn("ks_cobertura", r["disparadas"])

    def test_alarma_falsa_baja_en_ventanas_limpias(self):
        rng = np.random.default_rng(7)
        u = self.umbrales(300, rng)
        arr = datos_g(np.random.default_rng(8), 4000)
        alarmas = [M.alerta(self.ref.estadisticos(*M.ventana(rng, 300, *arr)), u)["alerta"] for _ in range(300)]
        self.assertLess(np.mean(alarmas), 0.10)


if __name__ == "__main__":
    unittest.main()
