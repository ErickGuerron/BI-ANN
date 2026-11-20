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
