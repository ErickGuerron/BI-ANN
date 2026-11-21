"""The fitted HistGradientBoosting is saved and reused, gives the same probabilities, and is refitted when the
recipe changes. Run: python -m unittest pruebas.test_gb_cache -v"""
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "nucleo"))
import inferencia as I  # noqa: E402

DS = "ds7_historico_garantias"


class GbCacheTest(unittest.TestCase):
    def setUp(self):
        self.meta = json.loads((I.AUX / f"{DS}.json").read_text(encoding="utf-8"))
        self.original = I.AUX
        self.tmp = tempfile.TemporaryDirectory()
        I.AUX = Path(self.tmp.name)
        self.X = pd.read_csv(I.V2_SALIDA / DS / "X_val_onehot.csv", float_precision="round_trip").fillna(0).head(300)

    def tearDown(self):
        I.AUX = self.original
        self.tmp.cleanup()

    def test_se_guarda_y_la_segunda_carga_da_las_mismas_probabilidades(self):
        ruta = I.AUX / f"gb_{DS}.joblib"
        self.assertFalse(ruta.exists())
        primero = I._ajustar_gb(DS, self.meta)
        self.assertTrue(ruta.exists())
        segundo = I._ajustar_gb(DS, self.meta)
        self.assertTrue(np.array_equal(primero.predict_proba(self.X), segundo.predict_proba(self.X)))

    def test_la_segunda_carga_no_reajusta(self):
        I._ajustar_gb(DS, self.meta)
        ruta = I.AUX / f"gb_{DS}.joblib"
        guardado = joblib.load(ruta)
        guardado["modelo"] = "marcador"
        joblib.dump(guardado, ruta)
        self.assertEqual(I._ajustar_gb(DS, self.meta), "marcador")

    def test_otra_receta_reajusta_y_reemplaza_el_archivo(self):
        I._ajustar_gb(DS, self.meta)
        otra = dict(self.meta, gb_iter=self.meta["gb_iter"] - 1)
        modelo = I._ajustar_gb(DS, otra)
        self.assertEqual(modelo.max_iter, otra["gb_iter"])
        self.assertEqual(joblib.load(I.AUX / f"gb_{DS}.joblib")["huella"]["iter"], otra["gb_iter"])

    def test_un_archivo_corrupto_se_reajusta(self):
        (I.AUX / f"gb_{DS}.joblib").write_bytes(b"no es un modelo")
        modelo = I._ajustar_gb(DS, self.meta)
        self.assertEqual(modelo.predict_proba(self.X).shape[0], len(self.X))


if __name__ == "__main__":
    unittest.main()
