"""Network sheet: data behind the "Ver red neuronal" modal (diagram, configuration, metrics).
Run: python -m unittest pruebas.test_ficha_red -v"""
import json
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))
import ficha_red  # noqa: E402
import visual_red  # noqa: E402

PROGRAMA = Path(__file__).resolve().parents[1]
TOP4 = ["ds7_historico_garantias", "ds1_ingreso_solicitudes", "ds6_gasto_provisiones", "ds5_pagos_multimes"]


class FichaRedTest(unittest.TestCase):
    def test_el_diagrama_coincide_con_los_pesos_guardados(self):
        for ds in TOP4:
            f = ficha_red.cargar_ficha(ds)
            pesos = json.loads((PROGRAMA / "resultados_convergencia" / f"pesos_{ds}.json").read_text(encoding="utf-8"))
            self.assertEqual(f["capas"], pesos["capas"], ds)
            self.assertEqual(len(f["entradas"]), f["capas"][0], ds)
            self.assertEqual(len(f["clases"]), f["capas"][-1], ds)
            for k, w in enumerate(f["coefs"]):
                self.assertEqual(np.array(w).shape, (f["capas"][k], f["capas"][k + 1]), f"{ds} capa {k}")

    def test_la_configuracion_es_la_de_la_red_de_la_app(self):
        esperada = {"ds1_ingreso_solicitudes": (10.0, 64), "ds5_pagos_multimes": (3.0, 128),
                    "ds6_gasto_provisiones": (10.0, 128), "ds7_historico_garantias": (10.0, 64)}
        for ds, (alpha, lote) in esperada.items():
            c = ficha_red.cargar_ficha(ds)["config"]
            self.assertEqual((c["alpha"], c["tamano_lote"]), (alpha, lote), ds)
            self.assertEqual(c["capas_ocultas"], [32], ds)
            self.assertTrue(c["sobremuestreo"], ds)

    def test_las_metricas_salen_del_ajuste_guardado(self):
        for ds in TOP4:
            ajuste = json.loads((PROGRAMA / "resultados_convergencia" / f"ajuste_{ds}.json").read_text(encoding="utf-8"))
            m = ficha_red.cargar_ficha(ds)["metricas"]
            self.assertEqual(m["prueba"]["exactitud_balanceada"], ajuste["prueba"]["exactitud_balanceada"], ds)
            self.assertEqual(m["validacion"]["auc_macro_ovr"], ajuste["validacion"]["auc_macro_ovr"], ds)
            self.assertEqual(m["cv_promedio"], ajuste["cv_promedio"], ds)

    def test_la_matriz_de_confusion_suma_los_casos_de_prueba_y_el_recall_cuadra(self):
        for ds in TOP4:
            f = ficha_red.cargar_ficha(ds)
            cm = np.array(f["confusion"])
            n = len(pd.read_csv(PROGRAMA / "resultados_convergencia" / f"predicciones_prueba_{ds}.csv"))
            self.assertEqual(int(cm.sum()), n, ds)
            self.assertEqual(cm.shape, (len(f["clases"]), len(f["clases"])), ds)
            for k in range(len(f["clases"])):
                self.assertAlmostEqual(f["recall"][k], cm[k, k] / cm[k].sum(), places=6, msg=f"{ds} recall {k}")
                self.assertAlmostEqual(f["precision"][k], cm[k, k] / cm[:, k].sum(), places=6, msg=f"{ds} precision {k}")

    def test_la_exactitud_de_la_matriz_coincide_con_la_del_ajuste(self):
        for ds in TOP4:
            f = ficha_red.cargar_ficha(ds)
            cm = np.array(f["confusion"])
            self.assertAlmostEqual(np.trace(cm) / cm.sum(), f["metricas"]["prueba"]["exactitud"], places=3, msg=ds)

    def test_la_temperatura_solo_aparece_donde_se_aplica(self):
        t = {ds: ficha_red.cargar_ficha(ds)["temperatura"] for ds in TOP4}
        self.assertAlmostEqual(t["ds1_ingreso_solicitudes"], 0.709, places=3)
        self.assertAlmostEqual(t["ds7_historico_garantias"], 0.592, places=3)
        self.assertIsNone(t["ds5_pagos_multimes"])
        self.assertIsNone(t["ds6_gasto_provisiones"])

    def test_ds7_avisa_que_la_clasificacion_usa_el_ensamble(self):
        self.assertIn("ensamble", ficha_red.cargar_ficha("ds7_historico_garantias")["nota"].lower())
        for ds in ("ds1_ingreso_solicitudes", "ds5_pagos_multimes", "ds6_gasto_provisiones"):
            self.assertEqual(ficha_red.cargar_ficha(ds)["nota"], "", ds)


class EntrenamientoTest(unittest.TestCase):
    """Training-side metrics: error by split, train-test gap, limits and learning curve."""

    def ficha(self, ds):
        return ficha_red.cargar_ficha(ds)

    def ajuste(self, ds):
        return json.loads((PROGRAMA / "resultados_convergencia" / f"ajuste_{ds}.json").read_text(encoding="utf-8"))

    def test_el_error_de_validacion_y_prueba_coincide_con_el_ajuste_guardado(self):
        for ds in TOP4:
            e, a = self.ficha(ds)["entrenamiento"], self.ajuste(ds)
            self.assertAlmostEqual(e["error"]["val"], 1 - a["validacion"]["exactitud"], places=6, msg=ds)
            self.assertAlmostEqual(e["error"]["test"], 1 - a["prueba"]["exactitud"], places=6, msg=ds)

    def test_la_exactitud_balanceada_coincide_con_el_ajuste_guardado(self):
        for ds in TOP4:
            e, a = self.ficha(ds)["entrenamiento"], self.ajuste(ds)
            self.assertAlmostEqual(e["exactitud_balanceada"]["test"], a["prueba"]["exactitud_balanceada"], places=6, msg=ds)
            self.assertAlmostEqual(e["exactitud_balanceada"]["val"], a["validacion"]["exactitud_balanceada"], places=6, msg=ds)

    def test_la_brecha_es_error_de_prueba_menos_error_de_entrenamiento(self):
        for ds in TOP4:
            e = self.ficha(ds)["entrenamiento"]
            self.assertAlmostEqual(e["brecha"], e["error"]["test"] - e["error"]["train"], places=9, msg=ds)
            self.assertAlmostEqual(e["brecha_abs"], abs(e["brecha"]), places=9, msg=ds)

    def test_la_brecha_de_ds5_es_la_documentada(self):
        self.assertAlmostEqual(self.ficha("ds5_pagos_multimes")["entrenamiento"]["brecha"], 0.0309, places=3)

    def test_los_limites_se_evaluan_sobre_el_valor_absoluto(self):
        for ds in TOP4:
            e = self.ficha(ds)["entrenamiento"]
            self.assertEqual(e["cumple_5"], e["brecha_abs"] < 0.05, ds)
            self.assertEqual(e["cumple_3"], e["brecha_abs"] < 0.03, ds)
        self.assertTrue(self.ficha("ds5_pagos_multimes")["entrenamiento"]["cumple_5"])
        self.assertFalse(self.ficha("ds5_pagos_multimes")["entrenamiento"]["cumple_3"])

    def test_una_brecha_negativa_tambien_se_mide_en_valor_absoluto(self):
        e = ficha_red.evaluar_brecha(error_train=0.20, error_test=0.14)
        self.assertAlmostEqual(e["brecha"], -0.06)
        self.assertAlmostEqual(e["brecha_abs"], 0.06)
        self.assertFalse(e["cumple_5"])

    def test_el_historial_tiene_una_fila_por_epoca_entrenada(self):
        for ds in TOP4:
            f = self.ficha(ds)
            self.assertEqual(len(f["entrenamiento"]["historial"]), f["metricas"]["epocas_entrenadas"], ds)

    def test_la_perdida_de_entrenamiento_es_positiva_y_finita(self):
        for ds in TOP4:
            p = self.ficha(ds)["entrenamiento"]["perdida"]
            for parte in ("train", "val", "test"):
                self.assertTrue(np.isfinite(p[parte]) and p[parte] > 0, f"{ds} {parte}")

    def test_sin_los_datos_de_entrenamiento_la_ficha_sigue_funcionando(self):
        original = ficha_red.DATOS
        try:
            ficha_red.DATOS = Path("no_existe")
            f = ficha_red.cargar_ficha("ds1_ingreso_solicitudes")
        finally:
            ficha_red.DATOS = original
        self.assertIsNone(f["entrenamiento"])
        self.assertTrue(f["metricas"])


class SeparacionTest(unittest.TestCase):
    def test_una_red_pequena_conserva_la_separacion_por_defecto(self):
        self.assertEqual(ficha_red.separacion_capas([4, 3, 2]), 240)

    def test_una_red_alta_se_ensancha_y_es_monotona_con_las_entradas(self):
        a, b = ficha_red.separacion_capas([96, 32, 3]), ficha_red.separacion_capas([124, 32, 4])
        self.assertGreater(a, 240)
        self.assertGreater(b, a)

    def test_la_separacion_llega_al_html_del_lienzo(self):
        h = visual_red.html_lienzo([2, 3, 2], [np.ones((2, 3)), np.ones((3, 2))], modo="final", separacion=777)
        self.assertIn('"separacion": 777', h)


class LienzoFinalTest(unittest.TestCase):
    def html(self, modo):
        return visual_red.html_lienzo([2, 3, 2], [np.ones((2, 3)), np.ones((3, 2))], modo=modo, alto=300)

    def test_el_modo_final_no_dice_epoca_actual(self):
        h = self.html("final")
        self.assertIn("Pesos del modelo final", h)

    def test_los_modos_anteriores_no_cambian(self):
        h = self.html("entrenamiento")
        self.assertIn("Pesos de la época actual", h)
        self.assertIn("Contribución de cada conexión a este caso", h)


if __name__ == "__main__":
    unittest.main()
