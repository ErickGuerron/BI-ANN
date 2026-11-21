# Redes neuronales de ds1, ds5, ds6 y ds7: métodos, problemas, soluciones y limitaciones

Este documento explica, para cada una de las cuatro redes del programa, qué predice, cómo se construyó, qué problemas aparecieron, por qué cada solución funcionó y qué quedó sin resolver. Todas las cifras salen de archivos del repositorio (se indica cuál). Cuando algo no se midió, se dice.

Detalles que ya tienen su propio documento: calibración (`docs/calibracion.md`), ds5 en profundidad (`docs/ds5_mora_leve.md`), ds7 (`docs/ds7_ml_puro.md`, `docs/ds7_ensamble_incertidumbre.md`, `docs/ds7_garantia_cascada_monitoreo.md`) y límites generales (`docs/limitaciones.md`).

## 1. Cómo leer este documento

1. La sección 2 es el resumen de las cuatro redes en una tabla.
2. La sección 3 explica los conceptos y el método que comparten las cuatro. Conviene leerla primero: las secciones por red solo dicen qué cambió en cada una.
3. Las secciones 4 a 7 son una por red: qué predice, datos, método, resultados, problemas y soluciones, limitaciones.
4. La sección 8 reúne las limitaciones que afectan a todas.

## 2. Resumen

| | ds1 ingreso de solicitudes | ds5 pagos multimes | ds6 gasto de provisiones | ds7 histórico de garantías |
|---|---|---|---|---|
| Qué predice | Qué tan alta es la cuota frente a la capacidad de pago | Días de mora en el cierre del corte | Si la calificación de la cartera mejora, se mantiene o empeora | Si hay garantía y si cubre poco o mucho el monto |
| Clases | HOLGADA, MODERADA, AJUSTADA, SUPERA | AL_DIA, MORA_LEVE, MORA_31_90, MORA_MAYOR_90 | MEJORA, IGUAL, DETERIORA | SIN_GARANTIA, COBERTURA_BAJA, COBERTURA_ALTA |
| Partición | 70/15/15 aleatoria estratificada | Temporal por corte (2022) | Temporal por corte | 70/15/15 aleatoria estratificada |
| Filas (train / val / prueba) | 8.713 / 1.868 / 1.868 | 21.601 / 3.635 / 7.271 | 20.589 / 3.455 / 6.987 | 12.804 / 2.744 / 2.744 |
| Entradas de la red | 124 (variables + `GB_*` + 10 NCA) | 112 (variables + `GB_*`) | 99 (variables + 10 NCA) | 96 (variables + `GB_*`) |
| Red de la app (scikit-learn) | 1 capa de 32, alpha 10, lote 64 | 1 capa de 32, alpha 3, lote 128 | 1 capa de 32, alpha 10, lote 128 | 1 capa de 32, alpha 10, lote 64 |
| Exactitud balanceada en prueba | 0,824 | 0,753 | 0,819 | 0,752 |
| AUC macro en prueba | 0,959 | 0,887 | 0,956 | 0,962 |
| Clase más débil (recall) | AJUSTADA 0,73 | MORA_LEVE 0,57 | DETERIORA 0,73 | COBERTURA_ALTA 0,64 |
| Calibración (temperatura) | Sí, T = 0,709 | No (empeoraba) | No (empeoraba) | Sí, T = 0,592 |
| Modelo que usa la app hoy | Red scikit-learn | Red scikit-learn (sin proxies) | Red scikit-learn | Ensamble de 5 redes Keras con `Valor_Garantiza`, cascada de rechazo |

Fuentes: `resultados_convergencia/ajuste_<ds>.json`, `resultados_convergencia/pesos_<ds>.json`, `../<ds>/resumen_v4.json`, `docs/calibracion.md`. La exactitud balanceada de ds7 de la tabla es la de la red de base; el ensamble final se mide con macro F1 (sección 7).

## 3. Método común

### 3.1 Qué es cada red

Perceptrón multicapa (MLP) con activación ReLU en las capas ocultas y softmax en la salida. La salida es una probabilidad por clase y la clase predicha es la de mayor probabilidad. La app calcula la red con NumPy a partir de los pesos guardados en `resultados_convergencia/pesos_<ds>.json`; no necesita entrenar al arrancar.

### 3.2 Cadena de preparación de la entrada

Cada caso pasa por los mismos pasos en entrenamiento y en la app (`nucleo/inferencia.py`):

1. **Preprocesamiento** con `preprocesador.joblib` (de `datasets_multiclase_v2`): limpieza, variables derivadas, `log1p` en montos y one-hot en categorías.
2. **Variables `GB_*`** (solo ds1, ds5, ds7): probabilidades de un modelo de árboles (HistGradientBoosting) que se entregan a la red como entradas extra, junto con el margen, la entropía y una marca de "zona de duda". Es un apilamiento: la red recibe la opinión del árbol además de los datos.
3. **Estandarización** con media y escala de train.
4. **NCA de 10 componentes** (solo ds1 y ds6): proyección lineal supervisada que acerca casos de la misma clase. Se ajusta solo con train, con una muestra de 5.000 filas.
5. **Red MLP.**

Qué conjunto de entradas usa cada dataset se decidió con una regla fijada antes de mirar la prueba (`../cambios_v4.md`): se evaluaron cuatro conjuntos (originales; originales + `GB_*`; y los dos con NCA) y se eligió el de mayor exactitud balanceada en validación. Resultado: ds1 usa F1N, ds5 y ds7 usan F1, ds6 usa F0N.

### 3.3 Entrenamiento, paso a paso (`pipeline/ajuste_convergencia.py`, `nucleo/ann_core.py`)

- **Búsqueda en rejilla con validación cruzada.** 3 arquitecturas (32; 64-32; 128-64) × 7 valores de regularización L2 `alpha` (de 1e-4 a 10) × con o sin sobremuestreo = 42 configuraciones. Cada una se evalúa con validación cruzada estratificada de 5 pliegues solo sobre train (fase 1). Las 3 mejores se repiten con 3 semillas más para ver cuánto varían (fase 2). La prueba no interviene en ninguna decisión.
- **Parada temprana.** Se entrena época a época. Se detiene cuando la exactitud balanceada de validación no mejora más de 0,001 durante 10 épocas seguidas (máximo 200) y se conserva el modelo de la mejor época.
- **Sobremuestreo.** En train se duplican al azar filas de las clases pequeñas hasta igualar a la mayor. Solo se hace en train, nunca en validación ni prueba.
- **Optimizador.** Adam, tasa de aprendizaje 0,001. No se comparó con otros (pendiente 1 de `docs/pendientes.md`).

### 3.4 Por qué se usa exactitud balanceada

Las clases están muy desbalanceadas (ds6: 95 % IGUAL; ds7: 90 % SIN_GARANTIA). La exactitud simple premia a un modelo que siempre responde la clase mayoritaria. La exactitud balanceada es el promedio del recall de cada clase: todas las clases pesan igual. También se reporta macro F1 (promedio del F1 por clase) y AUC (qué tan bien ordena casos de una clase contra el resto, sin fijar un umbral).

### 3.5 Problemas y soluciones que se repitieron en las cuatro redes

**Problema A: sin sobremuestreo la red ignora las clases pequeñas.**
Evidencia: con la misma red de 32 neuronas y `alpha = 1e-4`, la exactitud balanceada de validación cruzada pasa de 0,793 a 0,807 en ds1, de 0,709 a 0,757 en ds5, de 0,684 a 0,723 en ds7 y de **0,454 a 0,786 en ds6** (`tabla_fase1` de cada `ajuste_<ds>.json`).
Por qué funciona: la pérdida se calcula sobre todas las filas. Si el 95 % son de una clase, equivocarse en las otras cuesta poco y la red aprende a responder siempre la mayoritaria. Al igualar los tamaños, cada clase aporta lo mismo a la pérdida.
Costo que se aceptó: duplicar filas hace que la red vea copias exactas. Por eso la búsqueda eligió regularización fuerte (siguiente punto).

**Problema B: con sobremuestreo, la red memoriza las filas duplicadas.**
La búsqueda eligió `alpha` alto en las cuatro redes (10 en ds1, ds6 y ds7; 3 en ds5). La regularización L2 penaliza pesos grandes y limita cuánto puede ajustarse la red a filas repetidas. Es una interpretación coherente con lo medido (la mejor configuración siempre combina sobremuestreo con `alpha` alto), no una prueba causal. La confirmación que sí existe es que la brecha entre entrenamiento y prueba quedó pequeña (ver cada sección).

**Problema C: la red alcanza su mejor punto en muy pocas épocas.**
La época elegida por validación en el modelo final fue 2 (ds1), 4 (ds5), 3 (ds6) y 1 (ds7), y el entrenamiento paró entre las 11 y 14 épocas. La validación cruzada de la fase 1 sugería épocas más tardías (11 a 20), así que la época de un solo entrenamiento es ruidosa. Se aceptó porque la selección usa solo validación y la parada temprana evita sobreajustar. Queda como limitación: una sola semilla define los pesos que usa la app.

**Problema D: las columnas `GB_*` de train y de prueba no tienen la misma distribución.**
En train las probabilidades del árbol son fuera de muestra (cada fila se predice con un modelo que no la vio). En validación y prueba vienen del modelo completo. Esa diferencia puede limitar a la red (advertencia de `../cambios_v4.md`). Solución parcial: se mantiene fuera de muestra en train, que es la forma correcta de evitar que la red confíe en un árbol que memorizó. Para ds5 se midió el efecto con rigor (sección 5).

**Problema E: la confianza de la red no es la probabilidad de acierto.**
Una red puede acertar mucho y aun así dar 90 % en casos que acierta 60 % de las veces. Solución: escalado por temperatura (una sola temperatura T ajustada en validación, que no cambia la clase predicha). Mejoró NLL, Brier y ECE en ds7 y ds1; en ds5 y ds6 no mejoró en prueba y no se aplicó (`docs/calibracion.md`). La regresión isotónica quedó descartada: en ds6 bajó la exactitud balanceada de 0,819 a 0,346 porque casi siempre predecía la clase mayoritaria.

## 4. ds1: ingreso de solicitudes

### Qué predice

Un tramo de la razón `Valor_Cuota / Capacidad_Pago`: HOLGADA hasta 0,25; MODERADA hasta 0,75; AJUSTADA hasta 1; SUPERA por encima de 1 (`datasets_multiclase_v2/preparar_mc.py`). La razón misma y la variable binaria original `Cuota_Supera_Capacidad` se excluyen de las entradas por construcción (columnas `Info_*` y la Y binaria nunca son variables).

### Método

Conjunto F1N: variables originales + `GB_*` + 10 componentes NCA = 124 entradas. Red de 1 capa de 32 neuronas, `alpha = 10`, lote 64, sobremuestreo, época final 2. Temperatura T = 0,709.

### Resultados en prueba

Exactitud 0,848; exactitud balanceada 0,824; AUC 0,959; validación cruzada 0,820 (cuatro semillas entre 0,818 y 0,821). La regresión logística sobre el mismo conjunto llega a 0,832. Con temperatura: NLL 0,485 → 0,444, ECE 0,094 → 0,033, sin cambiar la exactitud.

Matriz de confusión en prueba (filas reales, columnas predichas):

| Real \ Predicha | HOLGADA | MODERADA | AJUSTADA | SUPERA | Recall |
|---|---|---|---|---|---|
| HOLGADA (537) | 516 | 21 | 0 | 0 | 0,96 |
| MODERADA (936) | 60 | 767 | 103 | 6 | 0,82 |
| AJUSTADA (170) | 0 | 28 | 124 | 18 | 0,73 |
| SUPERA (225) | 1 | 6 | 41 | 177 | 0,79 |

### Problemas y soluciones

- Aplican A a E de la sección 3.5. Sin sobremuestreo la red tuvo 0,793 en validación cruzada; con sobremuestreo y `alpha` 10, 0,820.
- Se probaron tres tamaños de red; ganó la más pequeña con regularización fuerte, lo que indica que el problema no necesita capacidad extra.

### Limitaciones

- Los errores caen casi todos en clases vecinas (hay 1 caso entre HOLGADA y SUPERA, de 762). La clase se define por cortes de una razón continua en 0,25, 0,75 y 1, así que los casos cercanos a un corte se confunden. AJUSTADA es la más difícil (precisión 0,46: 103 casos MODERADA se predicen AJUSTADA). No se estudió si hay información en los datos que permita separar mejor esos casos.
- La regresión logística de v4 sobre el mismo conjunto obtuvo 0,832 de exactitud balanceada, un poco más que la red de la app (0,824). Esa logística no se reentrenó con sobremuestreo ni con la misma búsqueda, así que la comparación no es exactamente justa, pero no hay evidencia de que la red aporte una ventaja.
- La temperatura se ajustó con una validación aleatoria (no temporal) y sí mejoró en prueba.

## 5. ds5: pagos multimes

### Qué predice

Los días de mora de la operación en el cierre del corte: AL_DIA (0), MORA_LEVE (1 a 30), MORA_31_90 y MORA_MAYOR_90. `Info_Dias_Morosidad_Objetivo`, que define la etiqueta, coincide al 100 % con `Dias_Morosidad_Actual` de ds4 en el mismo corte (comprobado en 25.542 filas cruzables). Las entradas son características de los pagos hasta el corte (no incluyen la mora del corte).

### Método de la red de la app

Conjunto F1: variables + `GB_*` = 112 entradas. Red de 1 capa de 32, `alpha = 3`, lote 128, sobremuestreo. Partición temporal por fecha de corte (entrena con los cortes iniciales, valida y prueba con los posteriores) porque se quiere predecir hacia adelante. Sin NCA, sin proxies, sin temperatura.

### Resultados de la app en prueba

Exactitud 0,725; exactitud balanceada 0,753; AUC 0,887. Validación cruzada 0,756, validación 0,772. La brecha entre entrenamiento y prueba es de +3,09 puntos: cumple el criterio de 5 % pero no el de 3 % (`docs/pendientes.md`, punto 9).

| Real \ Predicha | AL_DIA | MORA_LEVE | MORA_31_90 | MORA_MAYOR_90 | Recall |
|---|---|---|---|---|---|
| AL_DIA (5.536) | 4.130 | 1.326 | 64 | 16 | 0,75 |
| MORA_LEVE (1.155) | 257 | 653 | 238 | 7 | 0,57 |
| MORA_31_90 (492) | 4 | 55 | 411 | 22 | 0,84 |
| MORA_MAYOR_90 (88) | 2 | 2 | 8 | 76 | 0,86 |

Precisión por clase: AL_DIA 0,94; MORA_LEVE 0,32; MORA_31_90 0,57; MORA_MAYOR_90 0,63. El error dominante es AL_DIA predicho como MORA_LEVE (24 % de AL_DIA), y MORA_LEVE predicho como MORA_31_90 (21 % de MORA_LEVE).

### Problemas, soluciones y por qué funcionaron (rama `ann-ds5-mora-leve`)

Se estudió con Keras (misma arquitectura, variante B: entropía cruzada ponderada con raíz cuadrada del inverso de la frecuencia) porque permite pérdidas y pesos por clase personalizados. Punto de partida: F1 de MORA_LEVE 0,46.

| # | Problema | Qué se hizo | Resultado y por qué |
|---|---|---|---|
| 1 | MORA_LEVE tiene F1 bajo (0,46) | Multiplicar su peso (1 a 3) y ajustar un sesgo por clase sobre las probabilidades (`t12`, `t13`) | Sin mejora. El sesgo ajustado cambia de signo entre semillas. Es consistente con que AL_DIA y MORA_LEVE se solapan en las entradas (los casos de las dos clases se parecen), de modo que mover el umbral cambia qué errores se cometen pero no los reduce |
| 2 | No hay atraso por evento: `Dias_Atraso_Al_Pago` vale 0 en todas las filas | Derivar 9 proxies de puntualidad con las fechas de pago de los 6 meses anteriores (`t14`) | MORA_LEVE sube +0,037 de F1 (15 de 15 pares pliegue × semilla). Funciona porque quien paga siempre el mismo día se distingue de quien paga cuando puede, y eso no estaba en las entradas |
| 3 | Riesgo de fuga con las fechas de pago | Regla estricta (solo pagos anteriores al corte) y auditoría (`t15`): el AUC de `dia_desv` es 0,802 con `<=` el corte, 0,799 con `<` y 0,746 solo con pagos de meses anteriores | La señal no viene de pagos pegados al corte. Solo 621 de 32.507 filas tienen pago el día del corte |
| 4 | Una sola partición temporal es poco confiable | 3 pliegues temporales hacia adelante × 5 semillas (`t17`) | Mismo signo en los 3 pliegues |
| 5 | Las `GB_*` guardadas se generaron con la partición original, con información de cortes posteriores | Reajustar `GB_*` por pliegue (`t18`) | La base cae 4 puntos de macro F1 en los pliegues F1 y F2: estaban inflados. En F3, igual. La ganancia de los proxies se mantiene (+0,013 macro F1, +0,039 en MORA_LEVE) |
| 6 | ¿Hacen falta los 9 proxies? | Quitar uno a uno y probar subconjuntos (`t19`, `t20`) | Bastan 2: `dia_desv` y `dia_ult_menos_med` (MORA_LEVE 0,548 contra 0,544 con los 9). Los otros 7 no se distinguen del ruido |
| 7 | ¿Es la red o es el dato? | Regresión logística y HGB como referencia con las mismas entradas (`t24`) | Los tres modelos quedan entre 0,682 y 0,692 de macro F1: el techo parece de información, no de modelo |
| 8 | Casos externos (100 escenarios) | Evaluar acuerdo, no exactitud (`t21`, `t22`) | Las etiquetas son condicionales y sintéticas; no sirven para medir exactitud. Se detectó y corrigió un error propio: `Tasa_Interes = 15` entero se leía como categoría `OTROS`; ahora `inferencia.canonizar_categorias_numericas` normaliza `15`, `'15'` y `15.0` |
| 9 | Confusión de MORA_31_90 hacia MORA_LEVE (16 % a 25 % en los pliegues Keras) | Señal de cuotas atrasadas: cuotas esperadas por fecha de desembolso menos cuota pagada (`t25` a `t27`) | AUC 0,756 entre MORA_31_90 y MORA_LEVE, pero la red suma solo +0,008 de macro F1 y la confusión no baja (17,6 % a 17,1 % en F3). Con HGB la señal sumó más (+0,019 de macro F1, sobre todo en MORA_MAYOR_90) y la edad del crédito aportó otro punto; en la red la edad no aportó nada |
| 10 | ¿Mejora regularizar más? | Barrido de 12 configuraciones, regla de una desviación estándar, prueba leída una vez (`t28`) | Validación plana (0,734 a 0,741). La elegida (16 neuronas, alpha 10) rinde 0,696 contra 0,700 de la actual. La regularización no limita a la red |
| 11 | ¿Hay más datos para las clases confundidas? | Revisar `ds4` y `FACT_Cierre_Cartera_Mensual_por_operacion` | No: ds4 solo tiene mora de hasta 30 días, sus columnas de historial incluyen el corte o filtrarían la etiqueta, y solo 4.928 operaciones tienen etiqueta |

Resumen de lo medido con Keras (3 pliegues, 5 semillas, `GB_*` reajustado):

| Variante | Macro F1 | F1 MORA_LEVE |
|---|---|---|
| Sin proxies | 0,679 | 0,505 |
| Par de proxies | 0,692 | 0,548 |
| Par + cuotas atrasadas | 0,701 | 0,554 |

### Limitaciones que quedan

- **La app sigue con los pesos viejos de ds5.** Los proxies necesitan la tabla de pagos y el formulario no la tiene. Como el proyecto no saldrá a producción, no se resolvió de dónde saldrían. Con proxies, el modelo mejoraría unos 0,013 de macro F1 y 0,04 en MORA_LEVE.
- **Techo de información.** La confusión entre AL_DIA, MORA_LEVE y MORA_31_90 no se puede bajar con los datos disponibles: no hay días de atraso por evento ni historial de mora de las clases altas. MORA_LEVE queda cerca de 0,55 de F1.
- **Datos limitados:** una cooperativa y un solo periodo (2022); el macro F1 de prueba va de 0,65 a 0,72 según el pliegue. Los pliegues se solapan en entrenamiento y las mismas operaciones aparecen en varios cortes.
- **Cómo calculó el origen la etiqueta** (desde qué cuota se cuenta el atraso, pagos parciales, días de gracia) no se pudo verificar; queda pendiente con la cooperativa.
- **Sin calibración:** la temperatura de ds5 no mejoró en prueba, así que la confianza mostrada es la cruda.
- **Reajuste de `GB_*` no exacto:** el reajuste coincide en el argmax en 95,7 % de las filas con las columnas guardadas; no se encontró la causa.
- **MORA_MAYOR_90** tiene 88 casos en prueba y 99, 94 y 88 por pliegue: cualquier cifra de esa clase tiene un margen amplio y no se calcularon intervalos.
- **Caso documentado:** un crédito sin pagos (`Sin_Pagos = 1`) se clasifica AL_DIA con ~60 % de confianza; coincide con los datos (97,6 % de esos casos son AL_DIA) y se deja así (`docs/limitaciones.md`).

## 6. ds6: gasto de provisiones

### Qué predice

Si la calificación de la cartera de una operación mejora (MEJORA), se mantiene (IGUAL) o empeora (DETERIORA) entre un corte y el siguiente. Se excluyen las filas con calificación E, porque ya es la peor y no puede deteriorar (`preparar_mc.py`). Partición temporal por corte: 1 corte para validación, 2 para prueba.

### Método

Conjunto F0N: variables originales + 10 componentes NCA = 99 entradas (sin `GB_*`). Red de 1 capa de 32, `alpha = 10`, lote 128, sobremuestreo, época final 3. Sin temperatura.

### Resultados en prueba

Exactitud 0,848; exactitud balanceada 0,819; AUC 0,956. Validación cruzada 0,810, validación 0,842. Clases en prueba: IGUAL 94,9 %, MEJORA 2,8 %, DETERIORA 2,3 %.

| Real \ Predicha | MEJORA | IGUAL | DETERIORA | Recall |
|---|---|---|---|---|
| MEJORA (198) | 174 | 0 | 24 | 0,88 |
| IGUAL (6.630) | 224 | 5.634 | 772 | 0,85 |
| DETERIORA (159) | 38 | 5 | 116 | 0,73 |

Precisión por clase: MEJORA 0,40; IGUAL 0,999; DETERIORA 0,13.

### Problemas y soluciones

- **Desbalance extremo (95 % IGUAL).** Sin sobremuestreo la validación cruzada da 0,454 de exactitud balanceada; con sobremuestreo, 0,786, y con `alpha = 10`, 0,810. Es el caso donde el sobremuestreo hace la mayor diferencia, por la razón de la sección 3.5 (A).
- **Recall alto de las clases raras a costa de precisión.** De cada 100 casos que la red predice DETERIORA, solo 13 lo son. Es una consecuencia esperable de haber igualado las clases en entrenamiento: la red aprende como si las tres fueran igual de frecuentes, cuando DETERIORA es el 2,3 % de la prueba. No se definió cuánto cuesta cada tipo de error, así que no se decidió si ese equilibrio es el adecuado.
- **Calibración.** La temperatura de validación (T = 1,108) no mejoró en prueba y no se aplicó. La isotónica colapsó la exactitud balanceada a 0,346 (casi siempre predice IGUAL) y se descartó.
- **Validación y prueba temporales.** La exactitud balanceada de validación (0,842) es mayor que la de prueba (0,819). Una explicación probable es que los cortes posteriores se parecen menos al entrenamiento, y sería una razón por la que la temperatura ajustada en validación no se transfirió; no se comprobó.

### Limitaciones

- **Precisión baja de DETERIORA (0,13) y MEJORA (0,40).** Por la prevalencia de cada clase, aun un buen recall produce muchos falsos avisos. Para usarla en decisiones habría que fijar un umbral con el costo real de cada error; no se hizo.
- **Caso ds6-A sin resolver** (`docs/limitaciones.md`): con estado VENCIDO TOTAL, los datos dicen IGUAL en 70 % (n = 170), DETERIORA 22 % y MEJORA 8 %; la red predice DETERIORA con 40 %. El plan original del caso no quedó claro, y no se corrigió forzando la confianza.
- **El NCA se ajustó con una muestra de 5.000 filas** de train; no se midió cuánto cambia el resultado con otra muestra.
- **Una sola semilla** define los pesos de la app.
- **Sin experimentos adicionales.** Los estudios con Keras (proxies, ensamble, rechazo) se hicieron para ds5 y ds7; en ds1 y ds6 la red es la que salió de la búsqueda de scikit-learn y no se exploró más.

## 7. ds7: histórico de garantías

### Qué predice

SIN_GARANTIA si la operación no tiene garantía; si la tiene, COBERTURA_BAJA cuando `Valor_Garantiza / Valor_Entregado` es 2 o menos, y COBERTURA_ALTA si es mayor (`preparar_mc.py`). La etiqueta es una regla sobre la garantía y el monto.

### Evolución del modelo (cuatro etapas)

| Etapa | Qué es | Dónde |
|---|---|---|
| 1. Modelo base | Red scikit-learn con `Valor_Entregado` como sustituto de la cobertura (no ve la garantía) | `resultados_convergencia/pesos_ds7_*.json` |
| 2. ML puro | Entradas del formulario + `Valor_Garantiza` + `Valor_Entregado` en bruto, sin derivar la regla; MLP (64, 32), `alpha = 1e-3` | rama `ds7-garantia-capturada`, `docs/ds7_ml_puro.md` |
| 3. Ensamble con rechazo | 5 redes MLP; incertidumbre = desacuerdo entre redes; soporte por distancia a vecinos de entrenamiento | rama `ds7-ensamble-incertidumbre`, `docs/ds7_ensamble_incertidumbre.md` |
| 4. Versión de la app | Ensamble de 5 redes Keras (32 neuronas, ReLU, sobremuestreo, parada temprana) con `Valor_Garantiza` como entrada 97, cascada de dos pasos y monitoreo de drift | rama `ann-app-garantia-obligatoria`, `docs/ds7_garantia_cascada_monitoreo.md` |

### Resultados

**Red base** (prueba): exactitud 0,882; exactitud balanceada 0,752; AUC 0,962; con temperatura T = 0,592: NLL 0,419 → 0,356, ECE 0,130 → 0,049. Clases en prueba: SIN_GARANTIA 89,9 %, COBERTURA_BAJA 4,2 %, COBERTURA_ALTA 5,9 %. Recall: SIN_GARANTIA 0,91; COBERTURA_BAJA 0,71; COBERTURA_ALTA 0,64.

**Ensamble final con `Valor_Garantiza`** (prueba, una partición): macro F1 0,646 → **0,872**; recall de COBERTURA_BAJA / ALTA 0,548 / 0,642 → 0,757 / 0,951. Cascada: cobertura 91,4 % y exactitud de los aceptados 99,3 %.

Las cifras de las etapas no son comparables entre sí sin cuidado: cambian los datos de entrada, la receta, la métrica (error, exactitud balanceada, macro F1) y la partición. Este documento no las reconcilia.

### Problemas, soluciones y por qué funcionaron

| # | Problema | Qué se hizo | Resultado y por qué |
|---|---|---|---|
| 1 | La red base no puede ver la garantía, solo `Valor_Entregado`: confianza constante de 88 % y acierto de 7 % a 11 % en coberturas no vistas | Capturar `Valor_Garantiza` como entrada (campo obligatorio en la app) | Macro F1 de 0,646 a 0,872. La etiqueta es una regla sobre esa variable; sin ella la red solo podía adivinar por el monto |
| 2 | El modelo puro es seguro de sí mismo fuera de lo que vio: con garantía = 0,5 × monto acierta 9,5 % pero muestra 97 % de confianza | Medir el soporte (distancia a vecinos de entrenamiento en el subespacio garantía-monto, umbral en el percentil 99 de validación) | En k = 0,5 el error aceptado baja de 90 % a 1,6 %. Funciona porque ahí casi no hay ejemplos de entrenamiento (solo 8,1 % de las filas tienen garantía) y la distancia lo detecta |
| 3 | La calibración por temperatura no arregla ese exceso de confianza | Medir incertidumbre con un conjunto de 5 redes (información mutua = desacuerdo entre redes) | El error aceptado fuera de distribución baja de 98 %, 95 %, 91 %, 92 % a 0,8 %, 14,7 %, 9,7 %, 4,4 % (k = 0,5, 1,5, 2,5, 4). Cinco redes con semillas distintas coinciden en lo que aprendieron bien y discrepan donde la respuesta depende de la inicialización; ese desacuerdo marca casos poco aprendidos, mientras que la probabilidad media sigue alta |
| 4 | Un solo umbral de rechazo por confianza no tuvo efecto (quedó en su mínimo) | Cascada de dos pasos: filtro k-NN (k = 5) y entropía normalizada del ensamble con piso de cobertura por clase de 0,5 | Con `Valor_Garantiza` acepta el 91,4 % de los casos con 99,3 % de acierto entre los aceptados; contando las abstenciones como error, el macro F1 es 0,784. Se rechaza por "está lejos de lo visto" o "el ensamble duda", dos señales distintas |
| 5 | Aumento sintético (garantía = k × monto) para suplir los casos faltantes | Se probó y se descartó | +0,007 de macro F1 en validación, recall de ALTA −0,016 y pureza cruzada de ALTA −0,024 (supera el límite de 0,02). La ganancia no compensó el deterioro y se deshizo; no hay datos sintéticos en uso |
| 6 | Cambios de comportamiento de la garantía en el tiempo | Monitoreo de drift por ventanas: PSI, z-score de la proporción con garantía y KS de `log(garantía / monto)` | Detecta garantía ×10 con 300 casos (0,17 → 1,00) y ×2 (0,04 → 0,88). Solo funciona por lotes de al menos 300 casos |

### Limitaciones que quedan

- **Una partición y una semilla.** El ensamble se entrenó una vez; no hay intervalos.
- **La red aprende en buena parte la regla.** Como la etiqueta es una regla sobre la garantía y la garantía es entrada, un vector de ceros con garantía real se acepta al 100 %: las otras variables pesan poco.
- **COBERTURA_BAJA sigue débil con cobertura menor a 1 o cercana a 2:** casi no hay ejemplos y la masa de BAJA queda entre 1,4 y 1,9, pegada al umbral de 2. En 45 casos externos acierta 1 de 4 con cobertura 0,55 y 1 de 5 entre 1,75 y 1,95.
- **Dato sin aclarar:** el 49 % de las COBERTURA_BAJA de entrenamiento (377 de 769) tienen garantía en 0 con `Tiene_Garantia = 1`, y por monto se confunden con SIN_GARANTIA. Falta saber si es "sin dato" o "garantía real en 0".
- **La cascada no detecta un valor crudo extremo aislado** (`Anio_Desembolso = 9999` sube la aceptación a 92,5 %: el preprocesamiento lo neutraliza) ni cambios en la garantía (con garantía ×10 sigue aceptando el 94 %).
- **Confianza mostrada:** en los casos aceptados sigue entre 98 % y 99,7 % incluso en errores. La señal útil es el desacuerdo del ensamble, no la probabilidad media; la confianza debe leerse como estimación del modelo, no como probabilidad de acierto en casos nuevos.
- **Monitoreo no conectado** a la app; necesita ventanas de 300 casos, y las falsas alarmas en ventanas limpias fueron 4 % a 12 %. El asistente del formulario (`st.dialog`) no se probó de punta a punta.
- **Calibración del modelo base** se decidió después de ver la prueba (`docs/calibracion.md`): no es una estimación independiente.

## 8. Limitaciones que afectan a las cuatro redes

1. **Una sola semilla y una sola partición de prueba** para los pesos de la app. La búsqueda de hiperparámetros sí usó validación cruzada, pero los resultados de prueba no traen dispersión.
2. **No hay evidencia de que la red supere a un modelo lineal.** En `cambios_v4.md`, la regresión logística tuvo mejor exactitud balanceada que la red de la receta v4 en los 8 datasets (ds1 0,832 contra 0,814; ds5 0,741 contra 0,685; ds6 0,796 contra 0,730; ds7 0,735 contra 0,661). Las redes de la app, con la búsqueda de este programa (sobremuestreo y regularización), llegaron más alto que la receta v4 (ds5 0,753; ds6 0,819; ds7 0,752), pero la logística de v4 no se volvió a ajustar con ese procedimiento, así que esas cifras no son comparables entre sí. La única comparación hecha en igualdad de condiciones es la de ds5 (`t24`: mismas entradas, pliegues y semillas): la red y la logística empatan. Las redes sirven para ordenar casos (AUC 0,89 a 0,96).
3. **Los datos limitan más que el modelo.** En ds5 tres familias de modelos llegan al mismo nivel; en ds7 faltan ejemplos de cobertura media; en ds6 la clase de interés es el 2,3 % de la prueba.
4. **Generalización temporal.** ds5 y ds6 se parten por fecha. El rendimiento cambia entre pliegues (ds5: 0,65 a 0,72 de macro F1) y la validación no representa bien a la prueba; por eso la temperatura no se transfirió.
5. **Una sola cooperativa y un solo periodo** para ds5 (2022). No hay evidencia de que los resultados valgan para otra entidad o fecha.
6. **Optimizador no comparado.** Solo se usó Adam (pendiente 1 de `docs/pendientes.md`).
7. **Confianza.** Aun calibrada por temperatura, la confianza no es fiable fuera de lo visto en entrenamiento (ds7) y ds5/ds6 se muestran sin calibrar.
8. **El proyecto no saldrá a producción.** Las decisiones pendientes que dependían de un despliegue (de dónde salen los proxies de ds5, el monitoreo continuo de ds7) se descartaron o quedaron como sugerencias.

## 9. Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Búsqueda de hiperparámetros | `pipeline/ajuste_convergencia.py`, `resultados_convergencia/ajuste_<ds>.json` |
| Pesos de la red de la app | `resultados_convergencia/pesos_<ds>.json` |
| Predicciones de prueba | `resultados_convergencia/predicciones_prueba_<ds>.csv` |
| Calibración | `pipeline/calibrar.py`, `auxiliares/calibracion_<ds>.json`, `docs/calibracion.md` |
| ds5: experimentos `t12` a `t28` | `pipeline/t*.py`, `resultados_keras/t*.json`, `docs/ds5_mora_leve.md` |
| ds7: ensamble y cascada | `pipeline/t4_keras.py`, `pipeline/t8_valor_garantiza.py`, `nucleo/cascada.py`, `docs/ds7_*.md` |
| Casos de prueba con datos no vistos | `pruebas/casos_no_vistos.md`, `pruebas/casos_lejanos.md` |
| Definición de las etiquetas | `../../datasets_multiclase_v2/preparar_mc.py`, `../../datasets_multiclase_v2/LEEME.md` |
