# Calibración de las probabilidades de la red

## Por qué calibrar

La red entrega, para cada caso, una probabilidad por clase. La app muestra la mayor como **confianza**. Una red puede acertar mucho y aun así dar probabilidades poco fiables: por ejemplo, decir 90 % en casos que solo se aciertan el 60 % de las veces. Calibrar significa que, si el modelo dice p %, el caso se acierte cerca de p % de las veces.

Calibrar **no cambia la clase predicha** cuando se usa temperature scaling, y **no mejora la generalización** por sí solo. Lo que mejora es que la confianza sea informativa: un valor bajo debe significar que el modelo tiene poca evidencia.

## Dónde se hace

| Conjunto | Uso |
|---|---|
| Entrenamiento | Ajuste de los pesos de la red (ya hecho). |
| Validación | Ajuste de la temperatura T y de la regresión isotónica. |
| Prueba | Solo evaluación. No participa en ninguna decisión. |

Usar validación para calibrar evita que la confianza parezca buena por haberse ajustado a los mismos datos con que se mide.

## Temperature scaling

Es la técnica usada en la app.

1. La red produce logits `z` (la salida antes del softmax).
2. Se calcula `softmax(z / T)` con una sola temperatura `T > 0`.
3. `T` se elige para minimizar la pérdida logarítmica (NLL) en validación.

Efecto de T:
- `T > 1`: las probabilidades se acercan a la uniforme (la red baja su confianza).
- `T < 1`: las probabilidades se afilan (la red sube su confianza).
- Dividir todos los logits entre el mismo número **no cambia su orden**, así que la clase con mayor probabilidad no cambia.

Ventajas: un solo parámetro, pocos datos necesarios, no altera la clase predicha. Limitación: corrige una escala global, no los errores propios de cada clase.

## Regresión isotónica

Comparación, no se usa en la app.

1. Para cada clase `k`, se ajusta una función monótona no decreciente `f_k` que mapea la probabilidad `p_k` de la red al acierto observado en validación (1 si la clase real es `k`, 0 si no).
2. Para un caso nuevo se aplica `f_k` a cada probabilidad y luego se renormalizan para que sumen 1.

Ventajas: no asume una forma fija de la curva de calibración, así que puede corregir errores que la temperatura no corrige. Limitaciones: necesita más datos, puede sobreajustarse en las clases pequeñas, y como cada clase se calibra por separado puede **cambiar la clase predicha**. Para guardarla en la app haría falta almacenar una tabla por clase.

## Métricas

Todas se miden en **prueba**.

- **NLL (pérdida logarítmica):** penaliza fuerte la confianza alta en casos equivocados. Menor es mejor.
- **Brier:** error cuadrático entre la probabilidad y el resultado observado, promediado por clase. Menor es mejor.
- **ECE (error de calibración esperado):** divide los casos en 10 bins según su confianza y promedia la diferencia entre la exactitud y la confianza de cada bin, ponderada por el tamaño del bin. Menor es mejor. Cero significa calibración perfecta en promedio.
- **Exactitud y exactitud balanceada:** deben quedar iguales con temperature scaling. Si cambian, algo está mal.

## Proceso seguido

1. Se verificó que la entrada de la red (`X_*_nn.csv`) coincide con la dimensión de la primera capa.
2. `pipeline/calibrar.py` calcula los logits de validación y prueba con los pesos de `resultados_convergencia/pesos_<dataset>.json`.
3. Se ajusta `T` minimizando la NLL en validación.
4. Se ajusta la regresión isotónica en validación, solo para comparar.
5. Se miden las tres variantes (sin calibrar, temperatura, isotónica) en prueba.
6. Se guarda `T` en `auxiliares/calibracion_<dataset>.json`. `nucleo/inferencia.py` lee ese archivo y divide los logits entre `T` antes del softmax. Si el archivo no existe, `T = 1` y la app se comporta como antes.

## Resultados en prueba

Métricas en prueba. Exactitud balanceada = promedio del recall por clase.

| Dataset | T | Variante | NLL | Brier | ECE | Exactitud balanceada | Confianza media |
|---|---|---|---|---|---|---|---|
| ds7 garantías | 0,592 | sin calibrar | 0,419 | 0,195 | 0,130 | 0,752 | 0,758 |
| | | temperatura | **0,356** | **0,169** | **0,049** | 0,752 | 0,878 |
| | | isotónica | 0,171 | 0,091 | 0,014 | 0,698 | 0,926 |
| ds1 ingreso | 0,709 | sin calibrar | 0,485 | 0,250 | 0,094 | 0,824 | 0,754 |
| | | temperatura | **0,444** | **0,234** | **0,033** | 0,824 | 0,840 |
| | | isotónica | 0,377 | 0,195 | 0,016 | 0,808 | 0,872 |
| ds6 gasto | 1,108 | sin calibrar | 0,435 | 0,222 | **0,052** | 0,819 | 0,863 |
| | | temperatura | 0,433 | 0,221 | 0,061 | 0,819 | 0,842 |
| | | isotónica | 0,125 | 0,071 | 0,012 | **0,346** | 0,950 |
| ds5 pagos | 0,836 | sin calibrar | **0,651** | **0,382** | **0,031** | 0,753 | 0,716 |
| | | temperatura | 0,653 | 0,385 | 0,034 | 0,753 | 0,758 |
| | | isotónica | 0,466 | 0,266 | 0,017 | 0,730 | 0,815 |

Lectura:
- **ds7 y ds1:** la temperatura mejora NLL, Brier y ECE en prueba, sin cambiar la exactitud.
- **ds6 y ds5:** la temperatura no mejora en prueba. T se ajustó en validación, y en esos datasets la validación no representa bien la prueba (ds5 y ds6 son temporales). La diferencia es pequeña, pero empeora un poco en ECE (ds6) y NLL (ds5).
- **Regresión isotónica:** tiene el mejor NLL y ECE, pero en ds6 baja la exactitud balanceada de 0,819 a 0,346. Predice casi siempre la clase mayoritaria. No se usa.

Decisión final: temperature scaling **solo en ds7 y ds1**, donde mejora NLL, Brier y ECE en prueba. En ds6 y ds5 no se aplica, porque la temperatura ajustada en validación empeoró o no mejoró en prueba. Esta selección se hizo **después de ver los resultados de prueba**: no es una estimación independiente para ds6 y ds5. Una forma de evitar ese sesgo sería decidir con una partición de validación adicional, que no se hizo.

## Efecto en los casos de prueba (`pruebas/casos_no_vistos.md`)

| Caso | Etiqueta esperada | Antes | Después |
|---|---|---|---|
| ds5-A | AL_DIA | AL_DIA 59,6 % | AL_DIA 62,4 % |
| ds5-B | MORA_MAYOR_90 | MORA_31_90 70,1 % | MORA_31_90 76,6 % |
| ds6-A | IGUAL | DETERIORA 41,1 % | DETERIORA 40,5 % |
| ds7-B | SIN_GARANTIA | COBERTURA_BAJA 67,2 % | COBERTURA_BAJA 82,5 % |

Conclusiones:
- Calibrar no sube la confianza en los casos dudosos de forma útil. ds6-A sigue en 40 %, que es lo correcto.
- **Riesgo real:** en ds5-B y ds7-B la confianza en la clase equivocada subió. La calibración se ajusta a la validación; un caso fuera de distribución puede recibir una probabilidad alta aunque sea erróneo.
- La zona de duda de la app no usa esta probabilidad: se calcula con el modelo de gradient boosting (`GB_Zona_Duda`). Por eso no cambia con la calibración.

## Pendiente

- Reducir la confianza en casos fuera de distribución: por ejemplo, avisar cuando un valor clave esté lejos del rango de entrenamiento, o usar un modelo de incertidumbre.
- Eliminar `Valor_Entregado` como atajo en ds7 y reentrenar.
- Revisar la zona de duda basada en la red, en lugar de en el gradient boosting, si se quiere que la confianza y la zona de duda digan lo mismo.
