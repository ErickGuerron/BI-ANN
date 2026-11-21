# ds7: incertidumbre con conjunto de redes y rechazo

**Rama:** `ds7-ensamble-incertidumbre`
**Script:** `pipeline/ds7_ensamble_incertidumbre.py`

## Método

- Entradas: campos del formulario de ds7 + `Valor_Garantiza` + `Valor_Entregado` en bruto. Sin variables derivadas de la regla de la etiqueta.
- Conjunto de 5 redes MLP (64, 32), `alpha = 1e-3`, semillas 42 a 46, parada temprana sobre validación.
- Predicción: media de las probabilidades del conjunto.
- Incertidumbre epistémica: información mutua `I = H(media) - media(H(p_m))`. Mide el desacuerdo entre redes, no la confianza de cada una.
- Soporte: distancia media a los 10 vecinos de entrenamiento en el subespacio garantía-monto (log1p, estandarizado), con umbral `d* = 0,140` (percentil 99 de validación).
- Rechazo: "fuera de soporte" si `d > d*`; "indeterminado" si `I > u*`, con `u* = 0,0078` (percentil 95 de validación dentro de soporte). Solo se usan umbrales de validación; no hay etiquetas fuera de distribución.

La regla de la etiqueta solo se usa para calificar el sondeo.

## Criterio de aceptación

Brecha = error de prueba − error de entrenamiento del conjunto, menor a 3 pp.

| Error train | Error val | Error prueba | Brecha |
|---|---|---|---|
| 0,37 % | 0,73 % | 0,66 % | +0,28 pp (cumple) |

## Prueba (dentro de la distribución)

- Aceptados: 94,1 %, con precisión de 99,9 %.
- Indeterminados: 5,2 %.
- Fuera de soporte: 0,7 %.

## Sondeo fuera del dataset

Filas de prueba con garantía = k · monto, para k no visto en entrenamiento.

| k | Esperada | Aceptado correcto | Aceptado erróneo | Indeterminado | Fuera de soporte | Errores atrapados |
|---|---|---|---|---|---|---|
| 0 | SIN_GARANTIA | 90,7 % | 2,5 % | 6,6 % | 0,1 % | 72,9 % |
| 0,5 | COBERTURA_BAJA | 1,1 % | 0,8 % | 1,5 % | 96,6 % | 99,2 % |
| 1,5 | COBERTURA_BAJA | 8,9 % | 14,7 % | 30,3 % | 46,0 % | 83,8 % |
| 2,5 | COBERTURA_ALTA | 11,2 % | 9,7 % | 46,2 % | 32,9 % | 89,1 % |
| 4 | COBERTURA_ALTA | 39,8 % | 4,4 % | 28,3 % | 27,4 % | 92,7 % |

"Errores atrapados" = fracción de los casos que la red equivocaría que quedan rechazados.

Comparado con el modelo puro sin rechazo, el error aceptado fuera de la distribución baja de 98 %, 95 %, 91 % y 92 % (k = 0,5, 1,5, 2,5 y 4) a 0,8 %, 14,7 %, 9,7 % y 4,4 %.

## Lo que cumple y lo que no

- **Cumple** el criterio de brecha dentro de la distribución (+0,28 pp).
- **Cumple** el 3 % de error aceptado fuera de la distribución solo en k = 0,5 (0,8 %). En k = 1,5, 2,5 y 4 el error aceptado queda entre 4 % y 15 %, por encima del objetivo.
- **No cumple** que la confianza mostrada sea fiable: los casos aceptados mantienen 98 % a 99,7 % de confianza, también en los errores. La señal útil es el desacuerdo entre redes, no la probabilidad media.

## Limitaciones

- Una sola partición de prueba y cinco valores de k. No es una estimación robusta.
- Las coberturas altas (2,5 y 4) tienen ejemplos de entrenamiento, pero insuficientes para aprender la relación. El rechazo las detecta en parte; no las resuelve.
- La confianza que se muestre al usuario debe describirse como estimación del modelo, no como probabilidad de acierto en casos nuevos.
