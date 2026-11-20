# Pendientes y puntos de acción

Estado al cierre de la sesión. Rama de trabajo: `ds7-ensamble-incertidumbre`.

## Ramas

| Rama | Qué es |
|---|---|
| `modelo-base` | Versión actual del front (ds7 con Valor_Entregado como proxy). |
| `ds7-garantia-capturada` | Modelo puro con garantía capturada. Limitación documentada en `docs/ds7_ml_puro.md`. |
| `ds7-ensamble-incertidumbre` | Ensamble de 5 redes, soporte y rechazo. Documentado en `docs/ds7_ensamble_incertidumbre.md`. |

## Acciones pendientes

1. **Optimizador (no iniciado).** Adam ya es el optimizador por defecto de `MLPClassifier`. Falta comparar Adam, SGD con momento y L-BFGS con las mismas particiones, midiendo brecha de entrenamiento a prueba y el sondeo fuera de distribución. Hipótesis: no mejora el sondeo, porque el problema es de datos, no de optimización.
2. **Regularización (no iniciado).** Probar un `alpha` mayor y una red más pequeña, elegidos solo en validación. Medir brecha y sondeo.
3. **Datos reales en la zona de cobertura media (bloqueado por datos).** Faltan ejemplos con garantía entre 1 y 4 veces el monto. Es la única vía conocida para que el error aceptado fuera de distribución baje del 3 % en k = 1,5, 2,5 y 4.
4. **Varias particiones antes de declarar ganador.** Hoy todo sale de una sola partición de prueba y cinco valores de k.
5. **Criterio de confianza.** La confianza que se muestra a un caso aceptado no es probabilidad de acierto (sigue en 98 % a 100 % también en errores). Decidir cómo se presenta en el front.
6. **Datos de origen de ds7.** Revisar las 256 filas con etiqueta COBERTURA_BAJA y garantía nula o cero: distinguir "sin garantía" de "dato faltante".
7. **Decisión de modelo para la app.** Comparar base, puro y ensamble con los mismos criterios y elegir uno; hoy la app sigue con el modelo base.
8. **ds5-B y ds6-A (pendientes de definición).** ds5-B: la red ya usa GB_Esperanza como entrada, así que no hay otra red a la que volver. ds6-A: el plan original no quedó claro.
9. **ds5 brecha (+3,09 pp).** Cumple el 5 % pero no el 3 %. Revisar regularización o validación temporal.

## Decisión tomada

- Temperature scaling solo en ds7 y ds1. ds5 y ds6 quedaron sin calibración porque empeoraron en prueba (ver `docs/calibracion.md`).
- ds5-A queda como limitación documentada (ver `docs/limitaciones.md`).
