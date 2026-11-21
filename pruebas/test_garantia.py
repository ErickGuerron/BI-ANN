"""Tests for nucleo/garantia.py and the _garantia variant lookup of nucleo/cascada.py (temp files only).
Run: python -m unittest pruebas.test_garantia -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import cascada as C  # noqa: E402
import garantia as G  # noqa: E402


class TransformarTest(unittest.TestCase):
    def test_vacio_y_negativo_son_cero(self):
        np.testing.assert_allclose(G.transformar([np.nan, -50.0, 0.0]), [0.0, 0.0, 0.0])

    def test_es_log1p_de_montos_positivos(self):
        np.testing.assert_allclose(G.transformar([99.0]), [np.log(100.0)])


class AgregarTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(G, "AUX", Path(self.tmp.name))
        self.patch.start()
        G.parametros.cache_clear()
        self.g_train = np.array([0.0, 0.0, 1000.0, 5000.0, 20000.0])
        self.params = G.ajustar("dsx", self.g_train)

    def tearDown(self):
        self.patch.stop()
        G.parametros.cache_clear()
        self.tmp.cleanup()

    def test_el_conjunto_de_entrenamiento_queda_estandarizado(self):
        col = G.columna("dsx", self.g_train)
        self.assertAlmostEqual(float(col.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(col.std()), 1.0, places=4)

    def test_agregar_suma_una_columna_y_conserva_las_96(self):
        X = np.arange(2 * 96, dtype=float).reshape(2, 96)
        out = G.agregar("dsx", X, [0.0, 1000.0])
        self.assertEqual(out.shape, (2, 97))
        np.testing.assert_array_equal(out[:, :96], X)

    def test_monto_mayor_da_valor_mayor(self):
        col = G.columna("dsx", [0.0, 100.0, 100000.0])
        self.assertTrue(col[0] < col[1] < col[2])

    def test_disponible_segun_exista_el_archivo(self):
        self.assertTrue(G.disponible("dsx"))
        self.assertFalse(G.disponible("otro"))


class VarianteCascadaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.patch = mock.patch.object(C, "AUX", self.dir)
        self.patch.start()
        C.cargar.cache_clear()

    def tearDown(self):
        self.patch.stop()
        C.cargar.cache_clear()
        self.tmp.cleanup()

    def escribir(self, clave, usa_garantia):
        rng = np.random.default_rng(0)
        np.savez(self.dir / f"cascada_{clave}.npz", W0_0=rng.normal(size=(3, 2)), b0_0=np.zeros(2),
                 W1_0=rng.normal(size=(2, 3)), b1_0=np.zeros(3))
        np.save(self.dir / f"{clave}_train.npy", rng.normal(size=(50, 3)))
        cfg = {"miembros": 1, "k_vecinos": 3, "d_max": 100.0, "umbral_h": 1.0, "umbral_p": 0.0,
               "usa_garantia": usa_garantia, "train_npy": f"{clave}_train.npy"}
        (self.dir / f"cascada_{clave}.json").write_text(json.dumps(cfg), encoding="utf-8")

    def test_sin_variante_garantia_usa_los_archivos_base(self):
        self.escribir("dsx", usa_garantia=False)
        self.assertTrue(C.disponible("dsx"))
        self.assertFalse(C.usa_garantia("dsx"))

    def test_con_variante_garantia_tiene_precedencia(self):
        self.escribir("dsx", usa_garantia=False)
        self.escribir("dsx_garantia", usa_garantia=True)
        self.assertTrue(C.usa_garantia("dsx"))

    def test_evalua_con_la_matriz_de_entrenamiento_guardada(self):
        self.escribir("dsx_garantia", usa_garantia=True)
        r = C.evaluar("dsx", np.zeros((2, 3)))
        self.assertEqual(len(r), 2)
        self.assertAlmostEqual(sum(r[0]["probs"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
