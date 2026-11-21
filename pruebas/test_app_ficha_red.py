"""The "Ver red neuronal" button opens a modal with two tabs (Diagrama, Métricas) for each of the four networks.
Run: python -m unittest pruebas.test_app_ficha_red -v"""
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TOP4 = ["ds7_historico_garantias", "ds1_ingreso_solicitudes", "ds6_gasto_provisiones", "ds5_pagos_multimes"]


def texto(at):
    partes = [m.value for m in at.markdown]
    return "\n".join(partes)


class FichaAppTest(unittest.TestCase):
    def test_cada_red_tiene_su_boton_y_la_app_arranca_sin_errores(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        claves = {b.key for b in at.button}
        for ds in TOP4:
            self.assertIn(f"ver_red_{ds}", claves, ds)

    def test_el_boton_abre_el_modal_con_diagrama_y_metricas(self):
        for ds in TOP4:
            at = AppTest.from_file(APP, default_timeout=60).run()
            next(b for b in at.button if b.key == f"ver_red_{ds}").click().run()
            self.assertFalse(at.exception, f"{ds}: {[e.value for e in at.exception]}")
            etiquetas = [t.label for t in at.tabs]
            self.assertIn("Diagrama", etiquetas, ds)
            self.assertIn("Métricas", etiquetas, ds)
            t = texto(at)
            self.assertIn("Matriz de confusión en prueba", t, ds)
            self.assertIn("AUC de prueba", t, ds)

    def abrir(self, ds):
        at = AppTest.from_file(APP, default_timeout=60).run()
        next(b for b in at.button if b.key == f"ver_red_{ds}").click().run()
        self.assertFalse(at.exception, f"{ds}: {[e.value for e in at.exception]}")
        return at

    def test_las_metricas_de_entrenamiento_aparecen_en_cada_red(self):
        for ds in TOP4:
            t = texto(self.abrir(ds))
            for esperado in ("Error de entrenamiento", "Error de prueba", "Brecha |prueba − entrenamiento|",
                             "Pérdida (log-loss)", "Curva de entrenamiento"):
                self.assertIn(esperado, t, f"{ds}: falta {esperado}")

    def test_ds5_cumple_el_limite_de_5_pero_no_el_de_3(self):
        t = texto(self.abrir("ds5_pagos_multimes"))
        self.assertIn("3,09 pp", t)
        self.assertIn("Cumple el límite de 5 %", t)
        self.assertIn("No cumple el límite de 3 %", t)

    def test_ds1_con_error_de_prueba_menor_que_el_de_entrenamiento_muestra_signo_negativo(self):
        t = texto(self.abrir("ds1_ingreso_solicitudes"))
        self.assertIn("−0,61 pp", t)
        self.assertIn("Cumple el límite de 5 %", t)

    def test_hay_curvas_de_entrenamiento_por_red(self):
        for ds in TOP4:
            self.assertEqual(len(self.abrir(ds).get("vega_lite_chart")), 2, ds)  # st.line_chart: pérdida y exactitud

    def test_ds7_muestra_el_aviso_del_ensamble_y_ds1_no(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        next(b for b in at.button if b.key == "ver_red_ds7_historico_garantias").click().run()
        self.assertIn("ensamble de 5 redes", texto(at))
        at = AppTest.from_file(APP, default_timeout=60).run()
        next(b for b in at.button if b.key == "ver_red_ds1_ingreso_solicitudes").click().run()
        self.assertNotIn("ensamble de 5 redes", texto(at))


if __name__ == "__main__":
    unittest.main()
