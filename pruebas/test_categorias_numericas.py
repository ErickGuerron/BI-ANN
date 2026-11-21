"""A numeric category written in another form (15, '15', 15.0) is the same category, not OTROS.
Run: python -m unittest pruebas.test_categorias_numericas -v"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import inferencia as I  # noqa: E402

DS = "ds5_pagos_multimes"


class CategoriasNumericasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, M = I.motor()
        cfg = M.CONFIGS[DS]
        raw = pd.read_csv(Path(cfg["dir"]) / cfg["archivo"], float_precision="round_trip", low_memory=False,
                          dtype={cfg["id"]: str})
        cls.fila = raw.iloc[[5]].reset_index(drop=True)

    def probs(self, **cambios):
        f = self.fila.copy()
        for k, v in cambios.items():
            f[k] = v
        return I.predecir(DS, f)[1]

    def test_la_tasa_entera_equivale_a_la_tasa_decimal(self):
        base = self.probs(Tasa_Interes=15.0)
        for forma in (15, "15", "15.0", " 15 "):
            self.assertTrue(np.array_equal(base, self.probs(Tasa_Interes=forma)), repr(forma))

    def test_cargas_y_booleanos_numericos_en_otra_forma(self):
        base = self.probs(Cargas_Familiares=2.0, Sin_Pagos=0, Es_Persona_Juridica=0)
        otra = self.probs(Cargas_Familiares=2, Sin_Pagos="0", Es_Persona_Juridica=0.0)
        self.assertTrue(np.array_equal(base, otra))

    def test_una_columna_con_categoria_de_texto_tambien_se_normaliza(self):
        # Cargas_Familiares has '0.0'..'4.0' and also 'DESCONOCIDO'; 'DESCONOCIDO' must keep working
        self.assertTrue(np.array_equal(self.probs(Cargas_Familiares=3), self.probs(Cargas_Familiares=3.0)))
        self.assertTrue(np.array_equal(self.probs(Cargas_Familiares="DESCONOCIDO"),
                                       self.probs(Cargas_Familiares="desconocido")))
        self.assertFalse(np.array_equal(self.probs(Cargas_Familiares=3), self.probs(Cargas_Familiares="DESCONOCIDO")))

    def test_una_tasa_desconocida_sigue_siendo_otros(self):
        self.assertTrue(np.array_equal(self.probs(Tasa_Interes=16), self.probs(Tasa_Interes=99)))
        self.assertFalse(np.array_equal(self.probs(Tasa_Interes=16), self.probs(Tasa_Interes=15)))

    def test_un_texto_no_numerico_no_rompe(self):
        self.assertEqual(self.probs(Tasa_Interes="abc").shape, (1, 4))

    def test_columnas_de_texto_no_se_tocan(self):
        df = pd.DataFrame({"Sexo": ["MASCULINO"], "Tasa_Interes": [15]})
        out = I.canonizar_categorias_numericas(df, {"Sexo": ["MASCULINO", "FEMENINO", "OTROS"],
                                                    "Tasa_Interes": ["14.0", "15.0", "OTROS"]})
        self.assertEqual(out["Sexo"].iloc[0], "MASCULINO")
        self.assertEqual(out["Tasa_Interes"].iloc[0], "15.0")


if __name__ == "__main__":
    unittest.main()
