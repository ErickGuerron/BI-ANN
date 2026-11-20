Feature: ann-ds7-valor-garantiza. Rama: ann-ds7-valor-garantiza (desde ann-app-cascada-ds7, commit 305c706).
Objetivo: que la red de ds7 distinga COBERTURA_BAJA de COBERTURA_ALTA. Decisión del usuario: opción 1, agregar solo Valor_Garantiza como entrada (sin Tiene_Garantia y sin la razón garantía/monto). Sin push ni PR.
Ruta: inline. Riesgo: item 3 acotado (el resultado de ds7 en la app pasa a venir del ensamble de 97 entradas; el formulario de ds7 gana un campo opcional).

## Specs
S1. Valor_Garantiza es la entrada 97 de la red: log1p estandarizado con media y desviación de train; vacío o faltante = 0; negativo se recorta a 0.
S2. El formulario de ds7 pide Valor_Garantiza como campo opcional (vacío = 0). Los demás campos siguen siendo obligatorios.
S3. Se reentrena el ensamble de 5 redes (misma receta: 32 neuronas, mismas semillas, sobremuestreo) y se recalculan filtro k-NN y umbral de entropía solo en validación, con piso 0.5 de cobertura por clase.
S4. La cascada exportada no pisa los archivos de la versión anterior (sufijo _garantia) y reproduce el ensamble de Keras (paridad).
S5. Reevaluar con los 45 casos externos usando Valor_Garantiza_Supuesto como entrada, contra la red anterior.

## Tasks
T1 (S1, S4): nucleo/garantia.py; nucleo/cascada.py busca primero cascada_<ds>_garantia.* y usa la matriz de train guardada. Hecho. 9 pruebas.
T2 (S3, S4): pipeline/t8_valor_garantiza.py. Hecho. Paridad 2.4e-7, decisiones iguales 100 %.
T3 (S2): generar_campos.py agrega el campo; app.py acepta campos opcionales y usa el ensamble para ds7. Hecho. 5 pruebas; la app carga sin excepciones. No se ejercitó el flujo completo del asistente (st.dialog) con AppTest.
T4 (S5): pruebas/prueba_45_casos_ds7.py. Hecho.

## Log
L1 (2026-11-20): "1" = solo Valor_Garantiza.
L2 Diagnóstico previo (datos): la etiqueta es una regla (BAJA si Valor_Garantiza / Valor_Entregado <= 2, ALTA si > 2). 377 de 769 BAJA tienen garantía 0 con Tiene_Garantia = 1. Cobertura de BAJA con garantía > 0: p5 0.68, p25 1.42, p50 1.47, p95 1.92; ALTA: p5 2.2.
L3 Resultados T8 (resultados_keras/t8_resumen.json), ensamble de 5, una partición:
- Macro F1 de prueba 0.8715 (antes 0.6461); recall por clase SIN 0.987, BAJA 0.757, ALTA 0.951; ECE 0.0038.
- Cascada en prueba (umbral H 0.25, d_max 9.33): cobertura 91.4 % (por clase 0.951, 0.626, 0.543), macro F1 de aceptados 0.952, exactitud de aceptados 99.3 %, macro F1 con abstenciones como error 0.784 (antes 0.585).
L4 Resultados con los 45 casos (resultados_keras/prueba_45_casos_ds7_garantia.json):
- Ensamble con garantía: acuerdo indicativo 77.8 % (red anterior 73.3 %), macro F1 0.757 (0.711). Recall SIN 1.00, BAJA 0.40 (0.33), ALTA 0.93.
- Cascada: 62.2 % automático, 35.6 % ambigüedad, 2.2 % fuera de rango; acuerdo en los 28 automáticos 92.9 % (antes 77.1 %).
- BAJA por cobertura supuesta: 0 -> 2 de 3; 0.55 -> 1 de 4; 0.85 -> 1 de 2; 1.15 -> 1 de 1; 1.75-1.95 -> 1 de 5.
- Garantía en blanco en los 45 casos: ninguno se predice ALTA (26 SIN, 19 BAJA), 73.3 % automático. Vacío = 0 puede producir una clasificación automática equivocada si el usuario omite una garantía que sí existe.
- Limitación: en el entrenamiento casi no hay BAJA con cobertura < 1 y la masa de BAJA está en 1.4-1.9, pegada al umbral de 2; esos tramos siguen sin aprenderse.
