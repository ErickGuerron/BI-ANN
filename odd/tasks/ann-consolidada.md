Feature: ann-consolidada. Rama: ann-consolidada (merge de ann-ds7-valor-garantiza, que incluye ann-app-cascada-ds7, con ann-monitoreo-drift; sin conflictos). Sin push ni PR.
Objetivo: comparar con las mismas métricas y escenarios el modelo de ds7 sin y con Valor_Garantiza, cada uno con su cascada y su monitoreo recalibrado, y decidir qué línea continuar.
Ruta: inline. Riesgo: none (solo script de comparación y documentación).

## Specs
S1. Mismas métricas y mismos escenarios de drift (misma semilla) para el ensamble sin y con Valor_Garantiza.
S2. Umbrales de cascada y del monitoreo recalibrados con la validación de cada variante; prueba se mide una sola vez.
S3. Escenarios de drift sobre Valor_Garantiza para el modelo que la usa.

## Tasks
T1 (S1, S2): pipeline/t9_comparacion.py y resultados_keras/t9_comparacion.json. Hecho.
T2 (S3): escenarios garantia_x10, garantia_x0.1 y garantia_en_blanco. Hecho.

## Log
L1 (2026-11-20): "si puedes para poder sacar métricas reales y definir cuál continuamos".
L2 Calidad en prueba (ensamble de 5, una partición): sin garantía macro F1 0.646, recall BAJA 0.548, ALTA 0.642; con garantía macro F1 0.872, recall BAJA 0.757, ALTA 0.951; ECE 0.013 y 0.004.
L3 Cascada por caso en prueba: sin garantía cobertura 85.4 %, macro F1 aceptados 0.721, exactitud aceptados 96.0 %, macro F1 con abstenciones como error 0.585; con garantía 91.4 %, 0.952, 99.3 %, 0.784.
L4 Bajo drift de otras columnas, la exactitud de los aceptados con garantía se mantiene en 95-100 % (desplazamiento de 2 sd: 96.4 % frente a 83.4 % sin garantía). Con la garantía el modelo depende sobre todo de ese campo; un vector de ceros con garantía real se acepta al 100 %.
L5 Monitoreo, falsa alarma: sin garantía 3.0 % (n=300) y 3.3 % (n=1000); con garantía 1.3 % y 2.5 %. Deriva de covariables en columnas continuas: detección 100 % en ambas variantes. Cambio de proporciones de clase: mejor con garantía (n=1000, minorías x0.5: 0.97 frente a 0.54).
L6 Drift en Valor_Garantiza (solo con garantía): x10 -> cascada acepta 94.4 %; monitor n=300 12 % (completo), n=1000 94 %; x0.1 -> acepta 89.8 %; monitor 94 % y 100 %; en blanco -> acepta 92.5 %, ninguno ALTA; monitor 100 % y 100 %. Con 30 % de contaminación la detección es baja (1 % a 60 %). Solo ~8 % de las filas tienen garantía > 0, por eso el PSI de esa columna reacciona poco con ventanas chicas.
L7 45 casos externos: sin garantía acuerdo 68.9 %, macro F1 0.636, acuerdo en automáticos 77.1 %; con garantía 77.8 %, 0.757, 92.9 %.
L8 Limitaciones: una partición; escenarios sintéticos; el drift en Valor_Garantiza no cambia las etiquetas verdaderas en la simulación, por eso solo se mide cobertura y clase predicha.
