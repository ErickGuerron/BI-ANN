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

## Limitaciones

- La mejora en macro F1 es de alrededor de un punto. En MORA_LEVE es de unos 4 puntos, y la clase sigue en 0,50 a 0,57.
- Los pliegues comparten las columnas `GB_*` guardadas en los archivos v4; no se reajustaron por pliegue. Cada fila tiene un `GB_*` fuera de muestra respecto de sí misma, pero el modelo que lo generó se entrenó con la partición original.
- Los pliegues no son independientes: se solapan en entrenamiento y las mismas operaciones aparecen en varios cortes.
- MORA_MAYOR_90 (312 filas) tiende a bajar un poco con los proxies en la partición original; no se estudió.
- **La app no puede usar los proxies.** No recibe la tabla de pagos. Hace falta decidir si se precalculan en el sistema de la cooperativa o si la API los calcula al vuelo desde el historial de pagos. Hasta decidirlo, la app sigue con el modelo sin proxies.
- Los pesos de la app (`resultados_convergencia/pesos_ds5_*.json`) no se tocaron.
