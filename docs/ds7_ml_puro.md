# ds7 con garantía capturada: modelo de ML puro

**Rama:** `ds7-garantia-capturada`
**Archivos:** `pipeline/ds7_garantia_ml_puro.py`, `pipeline/ds7_rechazo.py`, `pipeline/reentrenar_ds7_garantia.py` (experimento con cobertura calculada, descartado para el modelo puro).

## Qué hace

- Entradas: campos del formulario de ds7 + `Valor_Garantiza` + `Valor_Entregado`, en bruto. **No** se calcula la cobertura ni ninguna otra variable derivada de la regla de la etiqueta.
- Excluidos: `Tiene_Garantia` (revela la clase) y `Valor_Avaluo` (no es un campo del formulario).
- Garantías faltantes se tratan como 0 (mismo criterio que la matriz de entrenamiento).
- Red: MLP (64, 32), `alpha = 1e-3`, parada temprana sobre validación.

## Resultados

| Medida | Valor |
|---|---|
| Error de entrenamiento | 0,54 % |
| Error de validación | 0,84 % |
| Error de prueba | 0,77 % |
| Brecha prueba − entrenamiento | +0,23 pp (cumple < 3 pp) |
| Exactitud balanceada, prueba | 0,953 |

## Limitación: fuera de la distribución de entrenamiento

Solo el 8,1 % de las filas de entrenamiento tienen garantía mayor que cero, y la cobertura de esas filas va del percentil 10 al 90 entre 1,43 y 11,89. Casi no hay ejemplos de cobertura baja (0,5 a 1,4).

Sondeo sobre filas de prueba con garantía = k · monto, para k no visto en entrenamiento. La regla solo califica la respuesta, no entra al modelo:

| k | Esperada | Acierto | Confianza media | Confianza cuando se equivoca |
|---|---|---|---|---|
| 0 | SIN_GARANTIA | 94,1 % | 98,4 % | 85,0 % |
| 0,5 | COBERTURA_BAJA | 9,5 % | 97,1 % | 98,1 % |
| 1,5 | COBERTURA_BAJA | 23,9 % | 92,4 % | 94,7 % |
| 2,5 | COBERTURA_ALTA | 34,4 % | 88,3 % | 90,6 % |
| 4 | COBERTURA_ALTA | 59,7 % | 93,4 % | 91,8 % |

Comparado con el modelo base en las mismas filas, el base tiene 88 % de confianza constante (no ve la garantía), con acierto fijo de 7 % a 11 % en las coberturas no vistas.

**Conclusión:** el modelo puro es estocástico (devuelve probabilidades), pero **no es fiable fuera de su soporte**. Su confianza es alta incluso cuando se equivoca, y la calibración por temperatura no lo corrige. La confianza solo debe presentarse como probabilidad de acierto dentro de la distribución de entrenamiento.

## Detección de soporte y rechazo (resultado parcial)

- Soporte medido en el subespacio garantía-monto (log1p, estandarizado), con umbral en el percentil 99 de validación (`d* = 0,140`). En prueba, 0,7 % queda fuera de soporte.
- En el sondeo, el error aceptado baja mucho donde el soporte es bajo: k = 0,5 pasa de 90 % a 1,6 %.
- Dentro de soporte (k = 2,5 y 4) el error se mantiene, porque la relación no está aprendida, no porque falte soporte.
- El rechazo por confianza no tuvo efecto (τ quedó en su valor mínimo, porque validación ya cumple la precisión objetivo).

## Próximo paso

Ver la rama `ds7-ensamble-incertidumbre`, que mide la incertidumbre con un conjunto de redes.
