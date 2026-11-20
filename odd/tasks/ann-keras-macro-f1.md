Feature: ann-keras-macro-f1. Rama: ann-keras-macro-f1 (desde ds7-ensamble-incertidumbre).
Alcance aprobado: opción 1, por etapas. ds5 y ds7 primero; después ds1-ds8. Sin push ni PR.
Ruta: inline (tarea pequeña por etapa). Riesgo: ninguno de los ítems de alto riesgo en S1-S2; S3+ instala TensorFlow (entorno).

## Specs
S1. Las gráficas de frontera deben trazar la red real, no un modelo sustituto. Regiones de decisión y entropía en el plano PCA de las entradas, resto de dimensiones en la media.
S2. Eliminar de graficos/ todo archivo que el generador no produce. Conservar solo red_ds5_pagos_multimes.png y red_ds7_historico_garantias.png. El borrado es irreversible (graficos/ no está en Git).
S3. Migrar las ANN a Keras/TensorFlow. Pendiente.
S4. Macro F1 y exactitud equilibrada por clase en todas las ANN. Parcial: S1 ya reporta ambas en validación.
S5. ds7: política de abstención (H normalizada > umbral o máximo < 0.65 -> REVISIÓN_MANUAL). Pendiente.
S6. ds5: entropía cruzada ponderada y focal loss, comparadas con la línea base. Pendiente.

## Tasks
T1 (S1): pipeline/graficar_fronteras_red.py. Hecho. Commit: pendiente.
T2 (S2): pipeline/limpiar_graficos.py, prueba y --aplicar. Hecho: 14 archivos borrados, 2 conservados.
T3 (S3): TensorFlow instalado en el entorno conda `bi` (Python 3.12.3): tensorflow 2.21.0, keras 3.15.1, solo CPU. Hecho.
T4 (S3-S6): migración y métricas. Pendiente.

## Log
L1 (2026-11-20): El usuario pidió opción 1 y, para las gráficas, que se corrija el dibujo y que el script borre las imágenes no usadas.
Observación: pesos_ds5 de la app (capa 32, alpha 3.0, 4 épocas) no coincide con receta_red_v4.json de ds5 (capas 64-32, alpha 0.1). Exactitud equilibrada de validación de la app para ds5: 0.7718; la receta v4 reporta 0.7166. Pendiente aclarar qué modelo es la referencia.
L2: El usuario confirma que el modelo de la app es una capa oculta de 32 neuronas. Equivalente en Keras: Input(112) -> Dense(32, relu) -> Dense(4, softmax), 3.748 parámetros en ds5. Input no es una capa con pesos, así que la red tiene 2 capas entrenables. Para ds7: Input(96) -> Dense(32) -> Dense(3), 3.203 parámetros (verificado en la prueba de instalación).
L3: Condición de migración: todas las ocultas usan activation="relu" explícito en Dense (ann_core.py:102 y inferencia.py:128). pesos_<ds>.json no guarda la activación; la app asume ReLU sin verificarlo. Las ocho redes usan ReLU en ocultas y softmax en salida.
L4: T4 iniciado. Replicación: pesos de la app cargados en Keras y comparados con el forward pass de NumPy. ds5: A = CE con sobremuestreo (réplica), B = CE ponderada sqrt sin sobremuestreo, C = focal (gamma 2) sin sobremuestreo; 3 semillas; selección por macro F1 de validación. ds7: conjunto de 5 redes Keras; política de abstención con cobertura mínima 70 % en validación; prueba se reporta una sola vez.
L5: T4 ejecutado con pipeline/t4_keras.py; resultados en resultados_keras/t4_resumen.json.
- Replicación: diferencia máxima de probabilidades 1.9e-7 (ds5) y 1.3e-7 (ds7); argmax idéntico.
- ds5 (macro F1 validación, media ± desv., 3 semillas): A CE+sobremuestreo 0.7095 ± 0.0056; B CE ponderada sqrt 0.7334 ± 0.0057 (elegida); C focal γ=2 0.7205 ± 0.0101. Prueba de B: macro F1 0.7044 ± 0.0017 (A 0.6698, C 0.6957). F1 por clase de B (semilla 42): AL_DIA 0.889, MORA_LEVE 0.463, MORA_31_90 0.712, MORA_MAYOR_90 0.744.
- ds7 ensamble de 5 redes: macro F1 validación 0.6625, prueba 0.6461. ECE prueba 0.013 (bien calibrado en el ensamble).
- ds7 abstención (umbral H_n = 0.33 elegido en validación con cobertura mínima 70 %): prueba cobertura 72.6 %, macro F1 de aceptados 0.800, exactitud de aceptados 99.1 %, pero macro F1 con abstenciones como error 0.400. Abstención por clase: SIN_GARANTIA 20.6 %, COBERTURA_BAJA 87.0 %, COBERTURA_ALTA 88.9 %.
- Hallazgo: el criterio de selección (máximo macro F1 de aceptados con cobertura ≥ 70 %) elige un umbral que se abstiene casi siempre en las clases minoritarias. Es un defecto del criterio, no de la red. Pendiente decidir un criterio por clase.
L6: Recomendación analizada: piso de cobertura por clase (pipeline/politica_abstencion.py, pruebas/test_politica_abstencion.py, 10 pruebas OK). Commit de respaldo previo: 2ada576.
- Viabilidad (umbral elegido en validación, prueba una vez): piso 0.3 -> H = 0.49, prueba cobertura 78.8 %, macro F1 aceptados 0.762, macro F1 con abstenciones como error 0.487. Piso 0.5 -> H = 0.62, cobertura 86.0 %, macro F1 aceptados 0.726, macro F1 con abstenciones como error 0.594. Piso 0.7 -> infactible en validación. Línea base sin política: macro F1 0.646.
- Simulación fuera de muestra (t5_resumen.json): la política NO protege contra entradas fuera del rango. Con entradas uniformes en [-4, 4] la cobertura es 72-81 % y la confianza media es 0.87. Con entradas normales de desviación 5 la cobertura es 74-81 %. Solo el vector cero se abstiene por completo.
- Escenarios etiquetados: con desplazamiento de 2 desviaciones en 20 % de las columnas, el macro F1 de aceptados cae a 0.57-0.60.
- Conclusión: el piso por clase es viable (0.3 y 0.5), pero no cumple la afirmación de proteger la generalización. El umbral por entropía no detecta entradas fuera del rango porque el ensamble es seguro en ellas. Pendiente decidir si se añade un detector de fuera de muestra (distancia al entrenamiento o información mutua del ensamble).
