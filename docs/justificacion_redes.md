# Justificación de las redes neuronales: sobreajuste, tamaño y reglas aplicadas

Este documento justifica las cuatro redes que muestra la app (ds1, ds5, ds6 y ds7) frente al riesgo de sobreajuste y frente a las reglas de tamaño de datos. Las cifras salen de `resultados_convergencia/`, `docs/redes_neuronales_metodos.md`, `cambios_v4.md` y de las funciones de `nucleo/ficha_red.py`. Lo que no se midió se dice explícitamente.

## 1. Qué responde cada red

Las preguntas son las de la carta de diseño (ver `nucleo/ficha_red.py`, `PREGUNTA_CARTA`):

| Red | Pregunta de la carta | Qué predice |
|---|---|---|
| ds1 ingreso de solicitudes | ¿Qué clientes presentan un perfil adecuado para recibir un nuevo crédito con un nivel de riesgo aceptable? | Tramo de la cuota frente a la capacidad de pago |
| ds5 pagos multimes | ¿Qué clientes son los que cumplen con el plazo de su crédito? | Tramo de mora al cierre: al día, 1–30, 31–90, más de 90 |
| ds6 gasto y provisiones | ¿Qué factores influyen en que un cliente cumpla o incumpla con el plazo de pago de su crédito? | Si la calificación mejora, se mantiene o empeora |
| ds7 histórico de garantías | ¿Qué clientes presentan un perfil adecuado para recibir un nuevo crédito con un nivel de riesgo aceptable? | Cobertura de la garantía: sin garantía, baja o alta |

Las redes ds2, ds3, ds4 y ds8 están entrenadas, pero no se muestran en el tablero.

## 2. Protocolo de datos y evaluación (común a las cuatro)

- **Partición.** ds1 y ds7: aleatoria estratificada 70/15/15. ds5 y ds6: temporal por corte de cierre. ds5 tiene 21.601 / 3.635 / 7.271 filas (train / val / prueba) y ds6 tiene 20.589 / 3.455 / 6.987.
- **Búsqueda solo con train y validación.** La búsqueda de arquitectura y regularización usa validación cruzada estratificada de **k = 5 pliegues**, aplicada solo sobre train (`pipeline/ajuste_convergencia.py`, `--folds` por defecto 5). La prueba no interviene en ninguna decisión de la búsqueda.
- **Prueba leída una vez.** La prueba se usa para reportar, no para elegir. La única excepción documentada es la decisión de calibrar ds7 y ds1 por temperatura, que se tomó viendo la prueba (ver §3.8).
- **Métricas.** Exactitud balanceada, macro F1 y AUC, porque las clases están muy desbalanceadas. Ver `docs/redes_neuronales_metodos.md`, §3.4.

## 3. Técnicas contra el sobreajuste

Estas son las técnicas que aplican a las cuatro redes, con su evidencia:

1. **Regularización L2 (`alpha`).** Búsqueda en rejilla de 1e-4 a 10, elegida por validación cruzada. La búsqueda eligió α alto en las cuatro redes (10 en ds1, ds6 y ds7; 3 en ds5). Penaliza pesos grandes y limita cuánto se ajusta la red a filas repetidas.
2. **Parada temprana.** Se detiene cuando la exactitud balanceada de validación no mejora más de 0,001 durante 10 épocas seguidas (máximo 200), y se conserva la mejor época. Las épocas finales fueron 2 (ds1), 4 (ds5), 3 (ds6) y 1 (ds7).
3. **Búsqueda con repetición de semillas.** Las tres mejores configuraciones se repiten con tres semillas más (fase 2) para ver cuánto varían. La validación cruzada entrega un rango por semilla en la app.
4. **Sobremuestreo solo en train.** Se duplican filas de clases pequeñas hasta igualar a la mayor. Es necesario para que la red aprenda las clases raras, pero **por sí mismo aumenta el riesgo de memorización**, porque la red ve copias exactas. Por eso la búsqueda eligió α alto en las cuatro redes. Las filas duplicadas se cuentan en el tamaño de train de la app; las filas únicas son menos.
5. **Estandarización y NCA ajustados solo con train.** Media, escala y proyección NCA (ds1 y ds6, muestra de 5.000 filas, semilla 42) salen de train. Validación y prueba se transforman con esos parámetros.
6. **Variables de árbol fuera de muestra en train (`GB_*`).** En ds1, ds5 y ds7, las probabilidades del árbol se calculan en train de forma fuera de muestra, para que la red no aprenda de un árbol que memorizó las filas que usa. Limitación: en validación y prueba vienen del modelo completo, así que la distribución no es idéntica (ver §6).
7. **Selección de entradas con regla fijada antes de ver la prueba.** Entre cuatro conjuntos de variables, se eligió el de mayor exactitud balanceada en validación (`cambios_v4.md`). Si otro quedaba a 0,005 o menos, se elegía el de mayor pureza.
8. **Calibración por temperatura, con criterio explícito.** La temperatura se ajusta en validación y se aplica solo donde mejora NLL, Brier y ECE en prueba: ds1 (T = 0,709) y ds7 (T = 0,592). En ds5 y ds6 no mejoró, y no se aplicó. El criterio se decidió viendo la prueba, así que no es una estimación independiente para ds5 y ds6 (`docs/calibracion.md`).
9. **Medición de la brecha entre entrenamiento y prueba.** Brecha = error de prueba − error de entrenamiento, con límites de 5 % y 3 %. Se mide con los pesos guardados, sin calibrar (`nucleo/ficha_red.py`).
10. **Evaluación temporal hacia adelante (ds5).** Tres pliegues temporales con cinco semillas dieron macro F1 de prueba entre 0,65 y 0,72 según el pliegue (`docs/ds5_mora_leve.md`). Es la comprobación más exigente de generalización que hay en el repositorio.
11. **Ensamble y rechazo (solo ds7 en la app).** La clasificación de la app usa cinco redes con semillas distintas. Su desacuerdo mide incertidumbre. Un filtro de soporte por distancia a vecinos de entrenamiento (k = 5) rechaza casos lejos de lo visto. Ver `docs/ds7_garantia_cascada_monitoreo.md`.

Lo que **no** se hizo, y se dice aquí para no inflar la justificación:
- No se comparó el optimizador. Se usó Adam con tasa 0,001 en todas las redes (`docs/pendientes.md`).
- Los pesos de la app vienen de una sola semilla. Las cifras de prueba no tienen dispersión entre semillas.

## 4. Por red

### 4.1 ds1 · ingreso de solicitudes

- **Entradas:** conjunto F1N, 124 entradas: variables originales, `GB_*` y 10 componentes NCA.
- **Arquitectura:** 1 capa oculta de 32 neuronas, α = 10, lote 64, sobremuestreo. Época final 2.
- **Filas:** 8.713 / 1.868 / 1.868.
- **Errores:** entrenamiento 15,82 %, validación 15,79 %, prueba 15,20 %. **Brecha −0,61 pp**: cumple el 5 % y el 3 %. La prueba resultó algo más fácil que el entrenamiento, no es una mejora real.
- **Resultados en prueba:** exactitud balanceada 0,824, AUC 0,959. La regresión logística sobre el mismo conjunto llega a 0,832, así que la red no aporta una ventaja medible sobre un modelo lineal.
- **Límite de tamaño:** la prueba tiene 1.868 filas. Hoeffding pide 4.612 para ±2 pp, así que **no alcanza ±2 pp**; su margen es de ±3,1 pp.

### 4.2 ds5 · pagos multimes

- **Entradas:** conjunto F1, 112 entradas: variables y `GB_*`.
- **Arquitectura:** 1 capa oculta de 32 neuronas, α = 3, lote 128, sobremuestreo. Época final 4.
- **Filas:** 21.601 / 3.635 / 7.271 (partición temporal).
- **Errores:** entrenamiento 24,43 %, validación 23,82 %, prueba 27,52 %. **Brecha +3,09 pp**: cumple el 5 % pero no el 3 %. El error alto ya en entrenamiento indica falta de información en las entradas, no memorización.
- **Resultados en prueba:** exactitud balanceada 0,753, AUC 0,887. MORA_LEVE tiene recall 0,57.
- **Límite de tamaño:** la prueba tiene 7.271 filas, que alcanzan ±2 pp (±1,6 pp), pero no ±1 pp.

### 4.3 ds6 · gasto y provisiones

- **Entradas:** conjunto F0N, 99 entradas: variables originales y 10 componentes NCA. Sin `GB_*`.
- **Arquitectura:** 1 capa oculta de 32 neuronas, α = 10, lote 128, sobremuestreo. Época final 3.
- **Filas:** 20.589 / 3.455 / 6.987 (partición temporal).
- **Errores:** entrenamiento 14,02 %, validación 16,38 %, prueba 15,21 %. **Brecha +1,19 pp**: cumple el 5 % y el 3 %.
- **Resultados en prueba:** exactitud balanceada 0,819, AUC 0,956. Precisión de DETERIORA 0,13: de cada 100 casos predichos como deterioro, 13 lo son. Es consecuencia de haber igualado las clases en train.
- **Límite de tamaño:** la prueba tiene 6.987 filas, que alcanzan ±2 pp (±1,6 pp), pero no ±1 pp.

### 4.4 ds7 · histórico de garantías (red base)

- **Entradas:** conjunto F1, 96 entradas: variables y `GB_*`. La red base no recibe `Valor_Garantiza`.
- **Arquitectura:** 1 capa oculta de 32 neuronas, α = 10, lote 64, sobremuestreo. Época final 1.
- **Filas:** 12.804 / 2.744 / 2.744.
- **Errores:** entrenamiento 11,09 %, validación 10,79 %, prueba 11,77 %. **Brecha +0,68 pp**: cumple el 5 % y el 3 %.
- **Resultados en prueba:** exactitud balanceada 0,752. La red base no ve la garantía, así que solo puede estimar la cobertura a partir del monto. La clasificación de la app usa el ensamble, con macro F1 de 0,872 en una partición.
- **Límite de tamaño:** la prueba tiene 2.744 filas. Hoeffding pide 4.612 para ±2 pp, así que **no alcanza ±2 pp**; su margen es de ±2,6 pp.

## 5. Tamaño de la red y reglas de datos

Las cifras salen de `nucleo/ficha_red.py` (`contraste_dimension`) y de la pestaña Justificación de la app.

| Red | Parámetros | Regla de 10 (10 × parámetros) | Filas de entrenamiento reales | ¿Cumple la regla de 10? | Filas de prueba | ¿Prueba alcanza ±2 pp (4.612)? |
|---|---|---|---|---|---|---|
| ds7 base | 3.203 | 32.030 | 12.804 | No (40 %) | 2.744 | No |
| ds1 | 4.132 | 41.320 | 8.713 | No (21 %) | 1.868 | No |
| ds6 | 3.299 | 32.990 | 20.589 | No (62 %) | 6.987 | Sí |
| ds5 | 3.748 | 37.480 | 21.601 | No (58 %) | 7.271 | Sí |

**Cómo se lee esto:**

- **La regla de 10 no se cumple en ninguna red.** Es una heurística de modelos lineales y árboles, no un umbral para redes neuronales. No se usa como justificación.
- **La justificación es la brecha, no el conteo de parámetros.** Las cuatro redes cumplen el límite de 5 %. Una red que sobreajusta tiene brecha grande, y estas no la tienen (salvo ds5 en el criterio de 3 %, que está documentado).
- **Hoeffding aplica a la prueba, no al entrenamiento.** Su cota dice cuántas filas de prueba hacen falta para un margen dado. Por eso se compara con las filas de prueba, y no se usa como mínimo de entrenamiento. Dos redes (ds1 y ds7) no alcanzan ±2 pp, y ninguna alcanza ±1 pp.
- **Los márgenes de Hoeffding son iguales para todas las redes que tienen el mismo tamaño de prueba.** No dependen de la red, sino de n, α = 0,05 y el margen buscado.

## 6. Limitaciones que afectan la justificación

1. **La validación cruzada de ds5 y ds6 es aleatoria estratificada, no temporal.** `StratifiedKFold` mezcla cortes de distintos meses dentro de los pliegues. La búsqueda de α y de arquitectura puede favorecer configuraciones que funcionan en meses cercanos entre sí. La única comprobación temporal son los pliegues de la investigación de ds5 (ver §3.10).
2. **Una sola semilla define los pesos de la app.** Las cifras de prueba no tienen intervalo.
3. **Las filas de prueba de ds1 y ds7 no alcanzan ±2 pp.** Los márgenes reportados son los que la prueba permite, no los que se podrían obtener con más datos.
4. **La distribución de `GB_*` no es idéntica entre train y validación/prueba** (ver §3.6). Puede limitar a la red, como advierte `cambios_v4.md`.
5. **Calibración decidida viendo la prueba** (ds1 y ds7). No se usa como estimación independiente.
6. **El sobremuestreo aumenta el riesgo de memorización.** Se mitiga con α alto y parada temprana; no se eliminó.
7. **ds5 sigue en 3,09 pp de brecha** y no cumple el criterio de 3 %. Ver `docs/ds5_mora_leve.md` para el cierre de la búsqueda de MORA_LEVE.

## 7. Otras reglas aplicadas

- **Sin fuga de la etiqueta.** Se excluyen las columnas `Info_*` y las Y binarias originales. ds6 excluye `Calificacion_Siguiente`, que es la fuente de la etiqueta. Se revisó que la calificación del mismo corte en ds6 es la etiqueta misma, y no se usa (ver `docs/limitaciones.md`, cierre de ds5).
- **Prueba sin decisiones.** La prueba no interviene en la búsqueda, en la selección de conjunto de entradas ni en la época de parada. La excepción de calibración está declarada en §3.8.
- **Datos fuera de la distribución.** En ds7, el ensamble y la cascada rechazan casos fuera de lo visto en entrenamiento. Los casos sintéticos de ds5 se usan para medir acuerdo con el escenario, no exactitud, porque sus etiquetas son condicionales.
- **Umbrales documentados antes de medir.** Brecha de 5 % y 3 %, y la regla de selección de conjuntos.
- **Reproducibilidad.** Los pesos, las predicciones de prueba, la búsqueda y la configuración están en `resultados_convergencia/`, y las funciones de métricas están en `nucleo/ficha_red.py` con pruebas en `pruebas/test_justificacion_ficha.py`.

## 8. Conclusión

Las cuatro redes tienen menos filas de entrenamiento que la regla de 10 por parámetro, y eso no basta para decir que sobreajustan. La evidencia de que no sobreajustan en la práctica es la brecha medida (todas dentro del 5 %), la regularización elegida por validación y la parada temprana. Las limitaciones que más pesan son el tamaño de la prueba en ds1 y ds7, la validación cruzada aleatoria en ds5 y ds6, y la brecha de ds5 en el criterio de 3 %.
