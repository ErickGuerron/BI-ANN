# Limitaciones documentadas

Casos en los que el modelo no alcanza la etiqueta esperada, o donde la confianza no es un indicador fiable. Cada una tiene su justificación técnica. No se corrigen forzando la confianza ni con reglas ad hoc.

## ds5-A: crédito sin pagos (`Sin_Pagos = 1`)

**Caso:** ningún pago registrado, 0 días desde el último pago, 48 cuotas. El modelo clasifica **AL_DIA** con confianza moderada (≈ 60 %, sin calibrar; ver `pruebas/casos_no_vistos.md`).

**Por qué se documenta y no se corrige**

- La clase coincide con los datos: en el conjunto de datos, `Sin_Pagos = 1` es AL_DIA en **97,6 %** de los casos (n = 1334). La expectativa original de mora alta era incorrecta, y se reemplazó por esta.
- La confianza moderada no proviene de la calibración: la temperatura de ds5 quedó fuera (ver `docs/calibracion.md`), y la calibración no subiría la confianza de todos modos.
- Técnicamente, un perfil con todos los campos de pago en cero es poco frecuente y comparte rasgos con casos de mora temprana. La red reparte probabilidad entre AL_DIA y MORA_LEVE, y esa distribución refleja incertidumbre del modelo en esa zona, no un error de cálculo.
- Forzar una confianza alta en ese perfil ocultaría una incertidumbre real. Lo honesto es mostrarla.

**Qué no sabemos todavía:** no medí cuántos casos de mora temprana tienen exactamente este perfil de cero pagos. Si se quiere confirmar la explicación, hay que contar esa combinación en los datos.

## ds6-A: vencido total con mora alta

Pendiente de definición. Ver la sección "Pendiente" de `docs/calibracion.md`.

## ds5-B y ds7-B

Pendientes de decisión (ver la sección "Pendiente" de `docs/calibracion.md`).

- **ds5 MORA_LEVE y MORA_31_90 (cierre 2026-11-21).** El F1 de MORA_LEVE queda cerca de 0,55 y MORA_31_90 se confunde con MORA_LEVE en 15 % a 25 % de los casos. Los datos no traen días de atraso por evento ni historial de mora de las clases altas; ds4 solo contiene mora de hasta 30 días. Ver `docs/ds5_mora_leve.md`.

## ds5: separación AL_DIA / MORA_LEVE (cierre de la búsqueda)

**Estado:** limitación aceptada. Los pesos de la app para ds5 no cambian. Se reproducen desde `X_*_nn.csv`: exactitud de prueba 0,725, igual a la documentación.

**Qué se probó y por qué no se adoptó:**

| Intento | Resultado en prueba | Decisión |
|---|---|---|
| Regla de decisión por costo esperado (AL_DIA↔LEVE más caro que 31_90↔MAYOR_90), elegida en validación | F1 AL_DIA 0,832 → 0,838; F1 LEVE 0,409 → 0,412; errores AL_DIA↔LEVE 1.583 → 1.535; errores de AL_DIA/LEVE con mora alta sin cambio (325 → 326) | No se adopta: la mejora es marginal y no reduce la confusión de la frontera |
| Mora del cierre anterior (ds4 en t−1) como entrada | AUC AL_DIA vs LEVE con regresión logística: 0,848 → 0,861 en prueba (+0,013) | No se adopta: requiere historial de mora por operación que la app no recibe |
| Calificación de ds6 en el mismo corte | Es la etiqueta misma (A → AL_DIA o LEVE; B/C → 31_90; D → MAYOR_90) | Fuga: descartada |
| Calificación de ds6 en el corte anterior | AUC 0,53 para AL_DIA vs LEVE | No aporta |
| Bandera de zona de duda (`GB_Zona_Duda`) | Error de la red 40,2 % dentro de la zona contra 21,7 % fuera; concentra 919 de 2.001 errores | Es una entrada de la red, no una regla de decisión. Usarla para abstenerse sería una política nueva, y aun así dejaría sin resolver 895 de 1.583 errores AL_DIA↔LEVE |

**Conclusión:** con las variables disponibles en el repositorio, el solape entre AL_DIA y MORA_LEVE es un límite de información. Reponderar clases, ajustar sesgos, regularizar y probar regresión logística o gradient boosting sobre las mismas entradas no lo redujeron. Lo único que mejoró la separación de forma consistente fue la puntualidad de pago (+0,04 de F1 en MORA_LEVE), y depende de datos de la tabla de pagos que la app no recibe.

**Qué cambiaría esta conclusión:** una variable de historial de atraso por operación, disponible al momento de la consulta, o una fuente de datos de pagos integrada a la app.
