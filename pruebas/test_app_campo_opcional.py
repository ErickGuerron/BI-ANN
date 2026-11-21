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


GARANTIA = {"nombre": "Valor_Garantiza", "tipo": "num", "min": 0.0, "max": 454955.0, "entero": True, "no_negativo": True,
            "ayuda": "Escriba 0 si la operaci\u00f3n no tiene garant\u00eda.",
            "aviso_cero": "Garant\u00eda declarada en 0: la operaci\u00f3n se clasifica como sin garant\u00eda."}


class CampoGarantiaObligatorioTest(unittest.TestCase):
    def test_vacio_es_un_error_que_explica_como_declarar_que_no_hay_garantia(self):
        for vacio in (None, ""):
            err, _, valor = app.revisar(GARANTIA, vacio)
            self.assertIsNotNone(err)
            self.assertIn("Escriba 0", err)
            self.assertIsNone(valor)

    def test_cero_escrito_es_valido_y_avisa(self):
        err, aviso, valor = app.revisar(GARANTIA, 0)
        self.assertIsNone(err)
        self.assertEqual(valor, 0.0)
        self.assertIn("sin garant", aviso)

    def test_un_monto_positivo_no_avisa(self):
        err, aviso, valor = app.revisar(GARANTIA, 8000)
        self.assertIsNone(err)
        self.assertIsNone(aviso)
        self.assertEqual(valor, 8000.0)

    def test_negativo_sigue_siendo_error(self):
        self.assertIsNotNone(app.revisar(GARANTIA, -1)[0])

    def test_la_ayuda_del_campo_muestra_como_declarar_que_no_hay_garantia(self):
        self.assertIn("Escriba 0", app.requisitos(GARANTIA))

    def test_un_campo_sin_ayuda_conserva_el_mensaje_de_siempre(self):
        err, _, _ = app.revisar(OBLIGATORIO, None)
        self.assertIn("falta el dato", err)
        self.assertNotIn("Escriba 0", err)

    def test_un_cero_en_otro_campo_no_genera_aviso(self):
        self.assertIsNone(app.revisar(OBLIGATORIO, 0)[1])

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "auxiliares" / "campos_ds7_historico_garantias.json").exists(),
                         "campos json not generated")
    def test_el_formulario_real_de_ds7_pide_la_garantia_como_obligatoria(self):
        import json
        campos = json.loads((Path(__file__).resolve().parents[1] / "auxiliares" / "campos_ds7_historico_garantias.json")
                            .read_text(encoding="utf-8"))
        c = next(x for x in campos if x["nombre"] == "Valor_Garantiza")
        self.assertFalse(c.get("opcional", False))
        self.assertIn("Escriba 0", c["ayuda"])
        self.assertTrue(c["aviso_cero"])


if __name__ == "__main__":
    unittest.main()
