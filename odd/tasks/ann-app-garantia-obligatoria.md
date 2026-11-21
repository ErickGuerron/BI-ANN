Feature: ann-app-garantia-obligatoria. Rama: ann-app-garantia-obligatoria (desde ann-monitoreo-garantia, commit e0e87f8). Sin push ni PR.
Objetivo: que en la app una garantía en blanco no se tome como 0 sin que el usuario lo decida.
Ruta: inline. Riesgo: item 3 acotado (cambia la validación del campo Valor_Garantiza del formulario de ds7: de opcional a obligatorio; no se toca ningún otro campo ni dataset).

## Specs
S1. Valor_Garantiza es obligatorio: vacío es un error que dice "Escriba 0 si la operación no tiene garantía."
S2. Un 0 escrito es válido y el resultado muestra el aviso "Garantía declarada en 0: la operación se clasifica como sin garantía."
S3. Un monto positivo no genera aviso; un negativo sigue siendo error.
S4. Los demás campos y datasets conservan su validación y sus mensajes.

## Tasks
T1 (S1-S4): app.py (_revisar_numero: ayuda y aviso_cero; requisitos muestra la ayuda), pipeline/generar_campos.py (el campo ya no es opcional y lleva ayuda y aviso_cero), pruebas/test_app_campo_opcional.py (13 pruebas). Hecho. RED observado (3 fallos y 1 error) y luego GREEN. La app carga sin excepciones. El soporte genérico de campos opcionales se conserva, sin uso en ds7.

## Log
L1 (2026-11-20): "1" = campo obligatorio o confirmación en la app. Se eligió campo obligatorio con 0 explícito: no agrega controles nuevos, es coherente con la validación de los demás campos y deja una decisión consciente del usuario.
L2 Motivo (rama ann-consolidada, L6 de la comparación): con la garantía en blanco ninguno de los 45 casos se predecía ALTA y el 73 % salía automático.
L3 Limitaciones: el flujo completo del asistente (st.dialog) no se ejercitó con AppTest; se probaron la validación, el texto de ayuda y el arranque de la app. Un 0 escrito por desconocimiento sigue siendo posible; el aviso en el resultado lo hace visible, pero no lo impide.
