# ds5: recuperar MORA_LEVE

Rama: `ann-ds5-mora-leve` (desde `ann-app-garantia-obligatoria`). Punto de partida: red Keras de una capa oculta de 32 neuronas, variante B (entropía cruzada ponderada con raíz cuadrada). F1 de MORA_LEVE en prueba: 0,46.

## Qué se probó

| Paso | Script | Resultado |
|---|---|---|
| a. Multiplicar el peso de MORA_LEVE (1,0 a 3,0) | `pipeline/t12_ds5_peso_leve.py` | Sin mejora. F1 de MORA_LEVE en prueba 0,43 a 0,47; el macro F1 no sube y AL_DIA baja. |
| b. Sesgo por clase sobre las probabilidades, ajustado en validación | `pipeline/t13_ds5_sesgo_clase.py` | Sin mejora. El sesgo ajustado cambia de signo entre semillas; los intervalos bootstrap cruzan el cero en 2 de 3 semillas. |
| c. Proxies de puntualidad de pago | `t14`, `t15`, `t16`, `t17` | Mejora real en MORA_LEVE (ver abajo). |

## Proxies de puntualidad

La tabla de pagos (`FACT_Pagos_Y_Recaudaciones_muestra`) trae fecha y cuota de cada movimiento, pero `Dias_Atraso_Al_Pago` vale 0 en todas las filas: no hay atraso por evento. Se derivan 9 proxies de las fechas de pago de los 6 meses previos al corte: desviación del día del mes de pago (`dia_desv`), huecos entre pagos (máximo, medio, desviación), número de fechas distintas, día del último pago y su diferencia con la mediana, fracción de cuotas pagadas en varias partes, y días desde el último pago frente al hueco medio. Más un indicador de dato faltante. Entradas de la red: 112 a 122.

- Regla de visibilidad: solo pagos con fecha estrictamente anterior al corte. Los nulos se imputan con la mediana de entrenamiento y todo se estandariza con estadísticas de entrenamiento.
- El mejor proxy individual es `dia_desv` (AUC 0,80 de MORA_LEVE contra AL_DIA en entrenamiento).

## Auditoría de fuga (`t15`)

El AUC de `dia_desv` es 0,802 con `<=` el corte, 0,799 con `<` el corte y 0,746 usando solo pagos anteriores al mes del corte. No cae al alejar los pagos del corte, así que la señal no viene de pagos cercanos a él. Solo 621 de 32.507 filas tienen pago el día del corte. Las filas de la tabla limpia y de los archivos de la red coinciden en el mismo orden. Limitación: no se pudo verificar cómo el sistema de origen calculó `Info_Dias_Morosidad_Objetivo`.

## Resultado en tres pliegues temporales, 5 semillas (`t17`)

ds5 se parte por fecha de corte, así que se usaron ventanas hacia adelante: F1 (entrena feb a may, valida jun, prueba jul y ago), F2 (feb a jun, jul, ago y sep) y F3 (la partición original). Diferencias en prueba, con proxies menos sin proxies:

| Pliegue | Macro F1 base → proxies | F1 MORA_LEVE base → proxies | Δ macro | Δ MORA_LEVE |
|---|---|---|---|---|
| F1 | 0,694 → 0,702 | 0,535 → 0,566 | +0,008 | +0,031 |
| F2 | 0,717 → 0,726 | 0,533 → 0,573 | +0,009 | +0,040 |
| F3 | 0,707 → 0,717 | 0,462 → 0,501 | +0,010 | +0,039 |
| Global (15 pares) | 0,706 → 0,715 | 0,510 → 0,546 | +0,009 ± 0,006 | +0,037 ± 0,008 |

MORA_LEVE mejora en los 15 pares pliegue por semilla. El macro F1 mejora en 14 de 15.

## `GB_*` reajustado por pliegue (`t18`, `t18b`)

Los pliegues de arriba usaban las columnas `GB_*` guardadas, que se generaron con la partición original. Se reajustaron por pliegue (HGB sobre las 101 columnas one-hot, probabilidades fuera de muestra con `GroupKFold` por corte en entrenamiento, modelo completo para validación y prueba). Mismas 5 semillas, mismos proxies.

| Pliegue | Base macro F1 (GB guardado → refit) | Δ macro con proxies (guardado / refit) | Δ MORA_LEVE (guardado / refit) |
|---|---|---|---|
| F1 | 0,694 → 0,652 | +0,008 / +0,008 | +0,031 / +0,029 |
| F2 | 0,717 → 0,679 | +0,009 / +0,012 | +0,040 / +0,040 |
| F3 | 0,707 → 0,706 | +0,010 / +0,018 | +0,039 / +0,049 |
| Global (15 pares) | 0,706 → 0,679 | +0,009 / **+0,013** | +0,037 / **+0,039** |

- **La preocupación era fundada.** Con `GB_*` reajustado, la base cae 4 puntos de macro F1 en F1 y F2. Esos pliegues estaban inflados por las columnas `GB_*` generadas con información de cortes posteriores. En F3, que es la partición original, no cambia (0,707 contra 0,706).
- **La ganancia de los proxies se mantiene y no es menor.** Con `GB_*` reajustado es +0,013 de macro F1 (positivo en 15 de 15 pares) y +0,039 en MORA_LEVE (15 de 15).
- **El reajuste no reproduce exactamente las columnas guardadas.** En F3, con las mismas entradas e hiperparámetros, la diferencia media de las probabilidades es de 0,001 a 0,09 según la columna (la mayor en `GB_Zona_Duda`, 0,09, y `GB_Margen`, 0,06), y el argmax coincide en el 95,7 % de las filas. No se encontró la causa (se probó variar el número de iteraciones). El macro F1 de la base en F3 sí coincide (0,706 contra 0,707), así que la calidad es equivalente, pero no es una réplica bit a bit.
- **Desviación de la receta:** F1 tiene solo 4 cortes de entrenamiento, así que su `GroupKFold` usa 4 pliegues en vez de 5. No se reajustaron los hiperparámetros por pliegue.
- Las probabilidades fuera de muestra de entrenamiento siguen usando cortes posteriores dentro del propio entrenamiento del pliegue (`GroupKFold` no respeta el orden temporal). No fuga prueba, pero el entrenamiento ve parte del futuro de sí mismo.

## Qué proxies hacen falta (`t19`, `t20`)

Mismo esquema que `t18` con `GB_*` reajustado: 3 pliegues temporales, 5 semillas, 15 pares. Base: macro F1 0,679, MORA_LEVE 0,505.

Quitar un proxy a la vez (`t19`) mostró que solo dos pesan: `dia_ult_menos_med` (día del último pago menos su día habitual) y `dia_desv` (variación del día de pago). Quitar cualquiera de los otros siete no se distingue del ruido. `dia_desv` solo, con un AUC individual de 0,80, aporta apenas +0,009 de los +0,039 en MORA_LEVE: el AUC por separado no predijo el aporte en la red.

| Variante (`t20`) | Macro F1 | F1 MORA_LEVE | Δ MORA_LEVE vs base | Pares donde supera a "todos" (MORA_LEVE) |
|---|---|---|---|---|
| Base (sin proxies) | 0,679 | 0,505 | — | — |
| Todos (9 proxies) | 0,692 | 0,544 | +0,039 | — |
| **Par: `dia_desv` + `dia_ult_menos_med`** | 0,692 | **0,548** | +0,043 | 10 de 15 |
| Trío: par + `dia_ult` | 0,693 | 0,545 | +0,040 | 10 de 15 |
| Cuatro: trío + `frac_cuotas_parciales` | 0,693 | 0,542 | +0,037 | 6 de 15 |

- **Dos proxies bastan.** El par iguala a los nueve (0,548 contra 0,544 en MORA_LEVE; 0,692 contra 0,692 en macro F1) y supera a la base en los 15 pares. Que supere a "todos" en 10 de 15 no es una diferencia demostrada: es compatible con un empate.
- **Agregar `dia_ult` o `frac_cuotas_parciales` no mejora el par.** El macro F1 sube a 0,693, por debajo de lo que distinguiría el ruido.
- **Consecuencia para producción:** bastan dos variables calculadas con las fechas de pago de los últimos 6 meses, en lugar de nueve.
- Ninguna variante cambia la confusión entre MORA_31_90 y MORA_LEVE: de los casos reales de MORA_31_90, entre el 19 % y el 21 % se predicen MORA_LEVE con cualquiera, y el recall de MORA_31_90 queda entre 0,749 y 0,763. Los proxies miden la irregularidad del día de pago, no cuántos días de atraso hay.

## Casos externos sintéticos (`t21`, `t22`)

Archivo `ANN_morosidad_100_orden_formulario_etiqueta.xlsx`: 100 escenarios, 25 por clase, 32 campos del formulario. Su hoja de instrucciones dice que las etiquetas son **condicionales y sintéticas, sin morosidad futura observada**, y pide no calcular exactitud real. Por eso se reporta acuerdo con el escenario, no exactitud. Se usan 99 casos (se excluye el 019, con campos vacíos; el 003 queda marcado: más cuotas pagadas que contratadas). Prueba la red que usa la app hoy, sin proxies: el formulario no trae fechas de pago.

| Escenario | → AL_DIA | → MORA_LEVE | → MORA_31_90 | → MORA_MAYOR_90 | Acuerdo |
|---|---|---|---|---|---|
| AL_DIA (24) | 14 | 10 | 0 | 0 | 58 % |
| MORA_LEVE (25) | 9 | 15 | 1 | 0 | 60 % |
| MORA_31_90 (25) | 0 | 0 | 24 | 1 | 96 % |
| MORA_MAYOR_90 (25) | 0 | 0 | 6 | 19 | 76 % |

- La confusión es entre AL_DIA y MORA_LEVE, en los dos sentidos. **Ningún caso de MORA_31_90 se predice como MORA_LEVE.** La confusión 31_90 → LEVE que se ve en los datos de prueba reales (16 % a 25 %) no aparece en este archivo.
- Los casos AL_DIA están dentro de la distribución: distancia al vecino más cercano de entrenamiento 0,93 (mediana), contra 1,8 en validación real; solo 1 de 24 supera el percentil 95. Sus 15 vecinos son 74 % AL_DIA y 23 % MORA_LEVE, coherente con la predicción. El solape entre AL_DIA y MORA_LEVE es real, no un problema de datos fuera de rango.
- El análisis de sustituir campos por valores típicos de AL_DIA no concluyó: ningún campo solo mueve más de 1 caso, y sustituir todos produce una fila mediana que no es un caso típico. No se usa.

**Corrección de un resultado propio.** Una primera versión de este análisis reportó que 23 de 24 casos AL_DIA se predecían MORA_LEVE. Era un error de preparación: se pasó `Tasa_Interes` y `Cargas_Familiares` como enteros crudos, y la red los leyó como categoría `OTROS`. La app convierte lo que se escribe (`15`) en el valor canónico (`15.0`) con `revisar`; el script se saltó ese paso. Ahora los casos pasan por el mismo `revisar` (`pipeline/casos_externos.py`). La función `predecir` de `nucleo/inferencia.py` tenía el mismo problema para cualquier consumidor que no fuera la app (scripts, una API): un valor numérico en otra forma se mandaba a `OTROS` sin error. Se corrigió en `onehot` (`canonizar_categorias_numericas`): `15`, `'15'` y `15.0` son ahora la misma categoría, las categorías de texto como `DESCONOCIDO` no se tocan y un número que no es ninguna categoría sigue siendo `OTROS`. Con el cambio, los 99 casos dan las mismas clases y probabilidades por los dos caminos (a través de la app y con enteros crudos).

## MORA_MAYOR_90: matriz de confusión por pliegue

Soporte en prueba: 99, 94 y 88 casos por semilla en F1, F2 y F3. Fracción de los casos reales de MORA_MAYOR_90 que se predicen en cada clase (suma sobre 5 semillas, `GB_*` reajustado):

| Pliegue | Variante | AL_DIA | MORA_LEVE | MORA_31_90 | MORA_MAYOR_90 |
|---|---|---|---|---|---|
| F1 | base | 0,230 | 0,022 | 0,349 | 0,398 |
| F1 | proxies | 0,240 | 0,016 | 0,352 | 0,392 |
| F2 | base | 0,223 | 0,049 | 0,236 | 0,491 |
| F2 | proxies | 0,219 | 0,053 | 0,240 | 0,487 |
| F3 | base | 0,023 | 0,030 | 0,173 | 0,775 |
| F3 | proxies | 0,023 | 0,032 | 0,130 | 0,816 |

- Los errores de MORA_MAYOR_90 van sobre todo a MORA_31_90 (13 % a 35 %). En F1 y F2 también a AL_DIA (22 % a 24 %), no en F3. Hacia MORA_LEVE solo van 2 % a 5 %, y los proxies no lo cambian: **no es un desplazamiento hacia MORA_LEVE.**
- El recall de MORA_MAYOR_90 no baja con los proxies (es igual en F1 y F2, y sube de 0,775 a 0,816 en F3). Lo que cambia es la **precisión**: en F3 pasa de 0,763 a 0,731, porque más casos reales de MORA_31_90 se predicen como MORA_MAYOR_90 (55 a 85 de 5 semillas). La caída de F1 de esa clase viene de ahí, no de perder casos graves.
- El soporte es de unos 90 casos por pliegue, así que estas fracciones tienen un margen amplio. No se calcularon intervalos.

## Modelos de referencia (`t24`)

Mismos 3 pliegues temporales, mismo `GB_*` reajustado, mismas variables, con y sin el par de proxies (`dia_desv` + `dia_ult_menos_med`). Regresión logística sobre las entradas de la red (peso por clase con raíz cuadrada; `C` elegido en validación entre 0,01, 0,1 y 1; salió 0,01 en la mayoría de los pliegues). HGB sobre las 101 columnas one-hot, con los hiperparámetros de v3 y sin ajuste. La red es el promedio de 5 semillas (`t20`). Prueba leída una vez por modelo.

| Modelo | Macro F1 medio sin proxies | Macro F1 medio con par | F1 MORA_LEVE con par | Macro F1 por pliegue con par (F1, F2, F3) |
|---|---|---|---|---|
| Red (Keras, variante B) | 0,679 | **0,692** | 0,548 | 0,661, 0,695, 0,722 |
| Regresión logística | 0,681 | 0,690 | 0,538 | 0,653, 0,690, 0,725 |
| HGB | 0,659 | 0,682 | **0,562** | 0,670, 0,676, 0,700 |

- **Los tres modelos quedan a menos de un punto de macro F1 entre sí** (0,682 a 0,692). La diferencia entre la red y la regresión logística (0,003) no se distingue del ruido entre semillas. La red no aporta una ventaja medible sobre un modelo lineal.
- **Los proxies ayudan a los tres:** macro F1 +0,013 (red), +0,009 (logística) y +0,022 (HGB). En MORA_LEVE: +0,043, +0,026 y +0,075. El efecto no depende de la arquitectura.
- **El techo parece de información, no de modelo.** Tres familias distintas llegan al mismo nivel con las mismas entradas.
- El HGB sin `GB_*` parte más abajo (0,659), y con proxies es el mejor en MORA_LEVE (0,562) pero no en macro F1.
- Una sola ejecución de logística y HGB por pliegue (son deterministas), así que sus cifras no traen dispersión entre semillas.

## Veredicto de cierre

**No está demostrado que la generalización y la exactitud actuales sean el máximo alcanzable.** Lo que sí está demostrado:

- Con lo ensayado, el peso por clase y el sesgo por clase no mejoran MORA_LEVE. Los proxies de puntualidad sí: +0,013 de macro F1 y +0,04 en MORA_LEVE, en 3 pliegues temporales y 5 semillas, con `GB_*` reajustado. Dos proxies bastan.
- Generalización medida: macro F1 de prueba entre 0,65 y 0,72 según el pliegue temporal (con `GB_*` reajustado, base: 0,652, 0,679 y 0,706 en F1, F2 y F3). Una sola cooperativa y un solo periodo (2022). No hay una prueba externa real: los dos archivos de casos sintéticos y el de Kiva no sirven para medir exactitud (ver arriba).
- Techo de información visible: MORA_LEVE se queda en torno a 0,50 a 0,55; AL_DIA y MORA_LEVE se solapan; MORA_31_90 se confunde con MORA_LEVE en 16 % a 25 % de los casos. Ninguna señal de las probadas mide cuántos días de atraso hay.

Lo que **no** se probó y puede mover los números:

- **Lo desplegado no es lo mejor que probamos.** Los pesos de ds5 de la app (capa de 32, alpha 3,0, sobremuestreo) no se tocaron. La variante B de Keras con los dos proxies no está en la app, y la app no puede calcular los proxies.
- **Había un modelo simple por encima de la red.** En `resumen_comparativo_v4.csv`, la regresión logística tuvo exactitud equilibrada de prueba 0,741 y la red de la app 0,685 (misma partición y conjunto de variables). La variante B de Keras llega a 0,74 a 0,75, y con proxies a 0,74 a 0,77. Eso sugiere que la receta de la app ya estaba lejos de su techo.
- Sin probar: comparar optimizadores (Adam, SGD con momento, L-BFGS), regularización (`alpha`, tamaño de capa), un ensamble de redes en ds5 (se usa en ds7), HGB y regresión logística con los proxies como referencia, y más datos reales de MORA_LEVE y MORA_31_90.

## Limitaciones

- La mejora en macro F1 es de alrededor de un punto. En MORA_LEVE es de unos 4 puntos, y la clase sigue en 0,50 a 0,57.
- Los pliegues no son independientes: se solapan en entrenamiento y las mismas operaciones aparecen en varios cortes.
- **La app no puede usar los proxies.** No recibe la tabla de pagos. Hace falta decidir si se precalculan en el sistema de la cooperativa o si la API los calcula al vuelo desde el historial de pagos. Hasta decidirlo, la app sigue con el modelo sin proxies.
- Los pesos de la app (`resultados_convergencia/pesos_ds5_*.json`) no se tocaron.

## Señal de cuotas atrasadas (t25, t26)

Se derivó, con `Fecha_Desembolso` de `FACT_Desembolsos` (cobertura 99,9 % de las filas), cuántas cuotas debían estar pagadas al corte (calendario mensual, tope `Numero_Cuotas`) y se restó la última cuota pagada con pagos anteriores al corte (`cuotas_atrasadas_max`).

- Por clase, la mediana de `cuotas_atrasadas_max` es 0 (AL_DIA), 0 (MORA_LEVE), 1 (MORA_31_90) y 3 (MORA_MAYOR_90). AUC MORA_31_90 contra MORA_LEVE: 0,756 (train y validación). Contar cuotas distintas pagadas en vez de la última da una señal peor (0,668).
- HGB en los 3 pliegues temporales, una semilla, prueba leída una vez (`t26`), macro F1 medio / F1 MORA_LEVE medio / % de 31_90 predicho como LEVE: sin proxies 0,659 / 0,487 / 16,6 %; con el par 0,682 / 0,562 / 17,5 %; par + `cuotas_atrasadas_max` 0,701 / 0,568 / 15,7 %; además `meses_desde_desemb` 0,710 / 0,574 / 15,1 %.
- La ganancia nueva viene sobre todo de MORA_MAYOR_90 (en un pliegue su F1 pasa de 0,72 a 0,81). MORA_LEVE casi no se mueve (+0,006) y la confusión 31_90 a LEVE baja solo 1 a 2 puntos. El techo de MORA_LEVE sigue siendo de información.
- `meses_desde_desemb` aporta por sí sola un punto más: puede ser edad del crédito y no atraso. Una semilla, un modelo; falta confirmar con la red y con más semillas.
### Confirmación con la red (t27)

Variante B, 3 pliegues temporales, 5 semillas, `GB_*` reajustado por pliegue (mismo protocolo que `t20`). Macro F1 / F1 MORA_LEVE / F1 MORA_31_90 / F1 MORA_MAYOR_90: sin proxies 0,679 / 0,505 / 0,704 / 0,621; par 0,692 / 0,548 / 0,707 / 0,616; par + `cuotas_atrasadas_max` 0,701 / 0,554 / 0,715 / 0,635; además `meses_desde_desemb` 0,701 / 0,552 / 0,714 / 0,637.

- La señal de cuotas suma unos 0,008 de macro F1 sobre el par y 0,006 en MORA_LEVE; MORA_31_90 sube 0,007 y MORA_MAYOR_90 casi 0,02. Con la red la ganancia es menor que con HGB.
- La edad del crédito no aporta en la red (0,7008 contra 0,7005), así que el punto extra de HGB era edad del crédito, no atraso.
- La confusión de MORA_31_90 hacia MORA_LEVE no cambia (pliegue F3: 17,6 % con el par, 17,1 % con la señal). El techo de MORA_LEVE sigue siendo de información.
- No se hizo un contraste formal entre variantes: son diferencias de unas décimas con 15 corridas dependientes (los pliegues se solapan).
### Barrido de regularización (t28)

12 configuraciones (capa 16, 32 o 64 neuronas; alpha 0,3, 1, 3 o 10), red con el par de proxies y `cuotas_atrasadas_max`, 3 pliegues temporales, 3 semillas, `GB_*` reajustado por pliegue. Selección por macro F1 de validación con la regla de una desviación estándar; la prueba se leyó una sola vez, solo para la elegida y para la configuración actual.

- La validación es plana: de 0,734 a 0,741 en las 12 configuraciones, con error estándar de 0,007 a 0,010. Todas caen dentro de una desviación estándar de la mejor (64 neuronas, alpha 1,0).
- La regla elige la más simple, 16 neuronas con alpha 10. En prueba: macro F1 0,696 y F1 MORA_LEVE 0,545, contra 0,700 y 0,554 de la configuración actual (32, alpha 3,0). No mejora.
- Conclusión: la regularización no limita a la red. Junto con la regresión logística (`t24`), confirma que el techo de ds5 es de información. No se cambian los pesos de la app.