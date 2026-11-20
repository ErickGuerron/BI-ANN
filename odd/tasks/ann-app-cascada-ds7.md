Feature: ann-app-cascada-ds7. Rama: ann-app-cascada-ds7 (desde ann-keras-macro-f1, commit a93d733).
Objetivo: probar la cascada (filtro k-NN fuera de rango + entropía del ensamble con piso 0.5) dentro de la app y decidir si sirve al objetivo del proyecto. Sin push ni PR.
Ruta: inline. Riesgo: item 3 acotado (se agrega una columna a la tabla del historial de todos los datasets; no se cambia ninguna salida existente).

## Specs
S1. La app muestra, para ds7, el estado de la cascada: AUTOMATICO, o REVISION_MANUAL con motivo FUERA_DE_RANGO o AMBIGUEDAD.
S2. Sin TensorFlow en ejecución: el ensamble de 5 redes se exporta a auxiliares/ y se evalúa con NumPy.
S3. Las decisiones exportadas deben coincidir con la política de referencia sobre validación.
S4. Simulación punta a punta por el camino de la app, con registros crudos reales y con campos crudos perturbados.
S5. Si cascada no está exportada para un dataset, la app se comporta como antes.

## Tasks
T1 (S2, S3): nucleo/cascada.py y pipeline/exportar_cascada_ds7.py. Hecho. Paridad con Keras 1.8e-7, decisiones idénticas 100 %.
T2 (S1, S5): app.py muestra el banner y una columna "Cascada" en el historial. Hecho. La app carga sin excepciones (streamlit AppTest).
T3 (S4): pruebas/simulacion_app_ds7.py y pruebas/test_cascada.py (6 pruebas; 24 en las tres suites). Hecho.

## Log
L1 (2026-11-20): "ambas" (cascada en la app y monitoreo de drift, cada una en su rama, para probar y decidir si es viable para el objetivo del proyecto).
L2 Resultados de la simulación (resultados_keras/simulacion_app_ds7.json, 600 registros de la prueba):
- Paridad entrada de la app vs X_test_nn.csv: diferencia máxima 5.0e-7.
- Registros reales: 88.0 % automático, 0.7 % fuera de rango, 11.3 % ambigüedad; exactitud de los automáticos 96.4 %.
- Perturbaciones en campos crudos: Valor_Entregado x3 -> 73.3 % automático; x50 -> 49.2 % automático, 16.7 % fuera de rango; Producto y Sucursal nuevos -> 18.3 % fuera de rango.
- Sin efecto o contrario: Anio_Desembolso = 9999 sube el automático a 92.5 %; Tasa_Interes = 99 87.5 %; Numero_Cuotas x5 88.0 %; Canton nuevo 88.0 %. El preprocesamiento neutraliza esos valores antes de la red, así que el filtro no los ve.
- Ejemplos extremos de pruebas/: "alto" y "bajo" -> REVISION_MANUAL por FUERA_DE_RANGO.
Hallazgo: un único campo extremo puede aumentar la aceptación automática; el filtro solo reacciona cuando varios campos se combinan o cambian categorías conocidas.
