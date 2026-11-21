# Pendientes y puntos de acción

Estado al cierre de la sesión. Rama de trabajo: `ds7-ensamble-incertidumbre`.

## Ramas

| Rama | Qué es |
|---|---|
| `modelo-base` | Versión actual del front (ds7 con Valor_Entregado como proxy). |
| `ds7-garantia-capturada` | Modelo puro con garantía capturada. Limitación documentada en `docs/ds7_ml_puro.md`. |
| `ds7-ensamble-incertidumbre` | Ensamble de 5 redes, soporte y rechazo. Documentado en `docs/ds7_ensamble_incertidumbre.md`. |
| `ann-ds5-mora-leve` | Trabajo de ds5 (MORA_LEVE y proxies de puntualidad) y cambios de la app (decimales, modelo GB guardado). Local, sin subir. Documentado en `docs/ds5_mora_leve.md`. |

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

## Actualización 2026-11-21 (ds5 MORA_LEVE)

Detalle en `docs/ds5_mora_leve.md`. Rama: `ann-ds5-mora-leve`.

- El peso por clase y el sesgo por clase no recuperan MORA_LEVE. Los proxies de puntualidad de pago, derivados de las fechas de la tabla de pagos, suben su F1 unos 4 puntos y el macro F1 cerca de 1 punto (tres pliegues temporales, 5 semillas).
- Cierra el punto 9 solo en parte: la brecha de entrenamiento a prueba no se estudió.
- Nuevos pendientes:
  1. Decidir de dónde saldrían los proxies en producción (precálculo o API). La app sigue sin ellos.
  2. ~~Reajustar `GB_*` por pliegue~~ Hecho (`t18`): la ganancia se mantiene (+0,013 macro F1, +0,039 MORA_LEVE) y la base de F1 y F2 estaba inflada unos 4 puntos. Queda por resolver por qué el reajuste no reproduce exactamente las columnas `GB_*` guardadas (concordancia del argmax: 95,7 %).
  3. ~~Revisar MORA_MAYOR_90~~ Hecho: el recall no baja; baja la precisión porque más casos de MORA_31_90 se predicen como MORA_MAYOR_90. Falta un intervalo de confianza (soporte de unos 90 casos por pliegue).
  3b. Ablación hecha (`t19`, `t20`): bastan 2 proxies, `dia_desv` y `dia_ult_menos_med`, para igualar a los 9 (MORA_LEVE 0,548 contra 0,544). Si se lleva a producción, calcular solo esos dos.
  3c. Confusión MORA_31_90 → MORA_LEVE (16 % a 25 % de los casos reales de 31_90). Los proxies no la tocan. Falta una señal sobre cuántos días de atraso hay, no solo cuándo paga.
  4. Verificar con el equipo de origen cómo se calcula `Info_Dias_Morosidad_Objetivo`.

### Dónde nos quedamos (2026-11-21, cierre de la sesión de ds5)

Estado: rama local `ann-ds5-mora-leve` (desde `ann-app-garantia-obligatoria`), 19 commits, el último `02bbf4e`. No está subida a origin. Trae también dos cambios de la app: los campos monetarios aceptan decimales, y el modelo de árboles se guarda en `auxiliares/gb_<ds>.joblib` en vez de reentrenarse al arrancar (arranque de 12 s a 5 s). `inferencia.py` normaliza las categorías numéricas (`15`, `'15'` y `15.0` son la misma).

Resultados (3 pliegues temporales, 5 semillas, `GB_*` reajustado por pliegue; detalle en `docs/ds5_mora_leve.md`):

- Peso por clase y sesgo por clase: sin mejora en MORA_LEVE.
- Proxies de puntualidad: +0,013 de macro F1 y +0,04 en MORA_LEVE. Bastan dos: `dia_desv` y `dia_ult_menos_med`.
- Referencias con el par de proxies (`t24`): red 0,692, regresión logística 0,690, HGB 0,682 de macro F1. La red no supera a un modelo lineal; el techo parece de información, no de modelo.
- Generalización: macro F1 de prueba entre 0,65 y 0,72 según el pliegue. Una cooperativa y un solo periodo (2022).
- Los archivos externos (`ANN_morosidad_100`, `UCI_100` híbrido, Kiva `loans_samp_0`) no sirven para medir exactitud: etiquetas condicionales o sintéticas, 67 de 100 perfiles UCI parten de filas de entrenamiento, y Kiva no tiene días de atraso ni los 32 campos.
- La app sigue con los pesos viejos de ds5 y no puede calcular los proxies.

Próximos pasos, en orden de prioridad:

1. Decidir de dónde saldrían `dia_desv` y `dia_ult_menos_med` en producción: precálculo por lotes (con la misma ventana de 6 meses) o cálculo en la API.
2. Verificar con el equipo de origen cómo se calcula `Info_Dias_Morosidad_Objetivo`.
3. Señal de magnitud del atraso (cuotas esperadas por fecha menos cuotas pagadas), la única ruta vista contra la confusión de MORA_31_90 y AL_DIA con MORA_LEVE. Necesita la fecha de desembolso.
4. Opcional: barrido pequeño de regularización (unas 12 configuraciones, regla de una desviación estándar, prueba leída una sola vez). Se espera una ganancia de décimas, no de puntos.
5. Una prueba externa real solo con cortes de la cooperativa posteriores a octubre de 2022.
6. Subir la rama y borrar `ann-app-montos-decimales` (ya fusionada), si se decide.

## Decisión tomada

- Temperature scaling solo en ds7 y ds1. ds5 y ds6 quedaron sin calibración porque empeoraron en prueba (ver `docs/calibracion.md`).
- ds5-A queda como limitación documentada (ver `docs/limitaciones.md`).

## Actualización 2026-11-20 (ds7 con Valor_Garantiza)

Estado y limitaciones en `docs/ds7_garantia_cascada_monitoreo.md`. Rama publicada: `ann-app-garantia-obligatoria`.

- Punto 7 (decisión de modelo para la app): ds7 usa el ensamble con `Valor_Garantiza`, con cascada y campo obligatorio. Macro F1 de prueba 0,872.
- Punto 6 (garantía en 0 con etiqueta BAJA): sigue abierto; 377 de 769 BAJA de entrenamiento.
- Nuevos pendientes:
  1. Pestaña "Monitoreo" por CSV en la app (sugerida; el monitoreo continuo espera a tener tráfico real).
  2. Más datos reales de `COBERTURA_BAJA` con cobertura menor a 1 o cercana a 2; el aumento sintético se probó y se descartó.
  3. Probar el asistente del formulario de punta a punta (`st.dialog` no se ejercitó).
  4. Repetir la comparación con más particiones y semillas.
  5. Consolidar y limpiar las ramas locales sin fusionar (`ann-keras-macro-f1`, `ann-app-cascada-ds7`, `ann-ds7-valor-garantiza`, `ann-monitoreo-drift`, `ann-monitoreo-garantia`, `ann-consolidada`).

### Actualización 2026-11-21 (decisiones y paso 3)

- Descartado: decidir de dónde saldrían los proxies en producción. El proyecto no saldrá a producción.
- Pendiente externo: cómo calcula el origen `Info_Dias_Morosidad_Objetivo` (preguntas: desde qué cuota se cuenta el atraso, si es el valor al corte o el siguiente mes, tratamiento del pago parcial, días de gracia).
- Hecho (`t25`, `t26`): señal de cuotas atrasadas con fecha de desembolso. Macro F1 de HGB 0,682 a 0,701 sobre el par de proxies (0,710 con la edad del crédito); MORA_LEVE casi igual (+0,006); gana sobre todo MORA_MAYOR_90. Detalle en `docs/ds5_mora_leve.md`.- Confirmado con la red (	27, 5 semillas): la señal de cuotas suma +0,008 de macro F1 sobre el par y +0,006 en MORA_LEVE; la edad del crédito no aporta. Sin efecto sobre la confusión 31_90 a LEVE.
- Hecho (	28): barrido de regularización. Validación plana (0,734 a 0,741); la configuración elegida por 1 SE (16, alpha 10) no mejora en prueba (0,696 contra 0,700). Sin cambios.
