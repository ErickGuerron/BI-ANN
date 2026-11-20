"""The optional numeric field (Valor_Garantiza): blank means 0, anything else is validated as before.
Run: python -m unittest pruebas.test_app_campo_opcional -v"""
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402

OPCIONAL = {"nombre": "Valor_Garantiza", "tipo": "num", "min": 0.0, "max": 454955.0, "entero": True,
            "no_negativo": True, "opcional": True}
OBLIGATORIO = {k: v for k, v in OPCIONAL.items() if k != "opcional"} | {"nombre": "Valor_Entregado"}


class CampoOpcionalTest(unittest.TestCase):
    def test_vacio_es_cero_sin_error(self):
        for vacio in (None, ""):
            err, _, valor = app.revisar(OPCIONAL, vacio)
            self.assertIsNone(err)
            self.assertEqual(valor, 0.0)

    def test_un_monto_valido_se_conserva(self):
        err, _, valor = app.revisar(OPCIONAL, 1500)
        self.assertIsNone(err)
        self.assertEqual(valor, 1500.0)

    def test_negativo_sigue_siendo_error(self):
        err, _, _ = app.revisar(OPCIONAL, -5)
        self.assertIsNotNone(err)

    def test_campo_obligatorio_vacio_sigue_siendo_error(self):
        err, _, valor = app.revisar(OBLIGATORIO, None)
        self.assertIsNotNone(err)
        self.assertIsNone(valor)

    def test_la_ayuda_del_campo_dice_que_es_opcional(self):
        self.assertIn("opcional", app.requisitos(OPCIONAL))
        self.assertNotIn("opcional", app.requisitos(OBLIGATORIO))


if __name__ == "__main__":
    unittest.main()
