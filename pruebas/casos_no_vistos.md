# Casos con datos que el dataset nunca vio y etiqueta esperada

Cómo usarlo: ingrese los valores del bloque en el asistente y compare la clase que muestra la app con la **etiqueta esperada**.

- **regla**: la etiqueta sale de la regla documentada del dataset.
- **datos**: la etiqueta es la mayoritaria observada en el dataset para casos parecidos (n indicado). Es una referencia estadística, no una regla: la clase real depende de información que el formulario no captura.
- Los campos no listados toman un valor típico del dataset (mediana o moda), para que el caso dependa solo de los campos clave.
- La verificación de 'no visto' es **por campo**: confirma que el valor clave no aparece en la columna, no que la fila completa sea nueva.

## ds1_ingreso_solicitudes

### ds1-A · etiqueta esperada **SUPERA** (regla)

Margen = Total_Ingresos - Total_Gastos = -1116 (≤ 0): la capacidad no alcanza y la cuota supera la capacidad.

Valores clave: Total_Ingresos = 4217, Total_Gastos = 5333, Monto_Solicitado = 18450.

Verificación por campo: Total_Ingresos: no aparece, Total_Gastos: no aparece, Monto_Solicitado: no aparece.

| Campo | Valor |
|---|---|
| Anio | 2015 |
| Total_Ingresos | 4217 |
| Total_Gastos | 5333 |
| Monto_Solicitado | 18450 |
| Mes | 1 |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 50+ |
| Sexo | FEMENINO |
| Estado_Civil | SOLTERO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | FAMILIAR |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | DESCONOCIDO |
| Total_Ingresos_Tramo | DESCONOCIDO |
| Patrimonio_Tramo | DESCONOCIDO |
| Es_Dependiente | 0.0 |
| Categoria_Credito | CONSUMO |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **SUPERA** (86.2 %), coincide con la etiqueta esperada.

### ds1-B · etiqueta esperada **HOLGADA** (regla)

Margen = 9158 y el monto es muy pequeño frente al margen: la razón cuota/capacidad queda por debajo de 0,25.

Valores clave: Total_Ingresos = 9871, Total_Gastos = 713, Monto_Solicitado = 310.

Verificación por campo: Total_Ingresos: no aparece, Total_Gastos: aparece, Monto_Solicitado: no aparece.

| Campo | Valor |
|---|---|
| Anio | 2015 |
| Total_Ingresos | 9871 |
| Total_Gastos | 713 |
| Monto_Solicitado | 310 |
| Mes | 1 |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 50+ |
| Sexo | FEMENINO |
| Estado_Civil | SOLTERO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | FAMILIAR |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | DESCONOCIDO |
| Total_Ingresos_Tramo | DESCONOCIDO |
| Patrimonio_Tramo | DESCONOCIDO |
| Es_Dependiente | 0.0 |
| Categoria_Credito | CONSUMO |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **HOLGADA** (99.0 %), coincide con la etiqueta esperada.

## ds5_pagos_multimes

### ds5-A · etiqueta esperada **AL_DIA** (datos)

En los datos, Sin_Pagos = 1 es AL_DIA en 97,6 % de los casos (n = 1334). Sin_Pagos no indica mora.

Valores clave: Sin_Pagos = 1, N_Movimientos = 0, N_Cuotas_Con_Pago = 0, Meses_Con_Pagos = 0, Pct_Cuotas_Pagadas = 0.0, Dias_Desde_Ultimo_Pago = 0, Numero_Cuotas = 48.

Verificación por campo: Sin_Pagos: aparece, N_Movimientos: aparece, N_Cuotas_Con_Pago: aparece, Meses_Con_Pagos: aparece, Pct_Cuotas_Pagadas: aparece, Dias_Desde_Ultimo_Pago: aparece, Numero_Cuotas: aparece.

| Campo | Valor |
|---|---|
| Valor_Entregado | 5703 |
| Numero_Cuotas | 48 |
| N_Movimientos | 0 |
| N_Cuotas_Con_Pago | 0 |
| Ultima_Cuota_Pagada | 11 |
| Monto_Total_Cobrado | 25866 |
| Monto_Promedio_Mov | 198.57 |
| Cuotas_Ult_3M | 3 |
| Cuotas_Ult_6M | 6 |
| Monto_Ult_3M | 2091 |
| Monto_Ult_6M | 4728 |
| Saldo_Ultimo_Pago | 3053 |
| Dias_Desde_Ultimo_Pago | 0 |
| Meses_Con_Pagos | 0 |
| Pct_Cuotas_Pagadas | 0.0 |
| Tasa_Interes | 10.0 |
| Sin_Pagos | 1 |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 50+ |
| Sexo | FEMENINO |
| Estado_Civil | SOLTERO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | FAMILIAR |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | BAJO (<500) |
| Total_Ingresos_Tramo | BAJO (<1000) |
| Patrimonio_Tramo | DESCONOCIDO |
| Es_Dependiente | 0.0 |
| Categoria_Credito | CONSUMO |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **AL_DIA** (59.6 %), coincide con la etiqueta esperada.

### ds5-B · etiqueta esperada **MORA_MAYOR_90** (datos)

En los datos, Dias_Desde_Ultimo_Pago ≥ 180 y Pct_Cuotas_Pagadas < 0,3 son MORA_MAYOR_90 en 100 % (n = 15: muestra pequeña).

Valores clave: Sin_Pagos = 0, Pct_Cuotas_Pagadas = 0.2, Dias_Desde_Ultimo_Pago = 200, Cuotas_Ult_3M = 0.

Verificación por campo: Sin_Pagos: aparece, Pct_Cuotas_Pagadas: aparece, Dias_Desde_Ultimo_Pago: aparece, Cuotas_Ult_3M: aparece.

| Campo | Valor |
|---|---|
| Valor_Entregado | 5703 |
| Numero_Cuotas | 36 |
| N_Movimientos | 129 |
| N_Cuotas_Con_Pago | 12 |
| Ultima_Cuota_Pagada | 11 |
| Monto_Total_Cobrado | 25866 |
| Monto_Promedio_Mov | 198.57 |
| Cuotas_Ult_3M | 0 |
| Cuotas_Ult_6M | 6 |
| Monto_Ult_3M | 2091 |
| Monto_Ult_6M | 4728 |
| Saldo_Ultimo_Pago | 3053 |
| Dias_Desde_Ultimo_Pago | 200 |
| Meses_Con_Pagos | 11 |
| Pct_Cuotas_Pagadas | 0.2 |
| Tasa_Interes | 10.0 |
| Sin_Pagos | 0 |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 50+ |
| Sexo | FEMENINO |
| Estado_Civil | SOLTERO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | FAMILIAR |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | BAJO (<500) |
| Total_Ingresos_Tramo | BAJO (<1000) |
| Patrimonio_Tramo | DESCONOCIDO |
| Es_Dependiente | 0.0 |
| Categoria_Credito | CONSUMO |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **MORA_31_90** (70.1 %), no coincide con la etiqueta esperada.

## ds6_gasto_provisiones

### ds6-A · etiqueta esperada **IGUAL** (datos)

Con Estado VENCIDO TOTAL en los datos: IGUAL 70 %, DETERIORA 22 %, MEJORA 8 % (n = 170). La clase mayoritaria es IGUAL.

Valores clave: Estado_Actual = VENCIDO TOTAL, Dias_Morosidad = 412, Calificacion_Actual = B.

Verificación por campo: Estado_Actual: aparece, Dias_Morosidad: no aparece, Calificacion_Actual: aparece.

| Campo | Valor |
|---|---|
| Dias_Morosidad | 412 |
| Saldo_Actual | 4271 |
| Saldo_Vencido | 0 |
| Valor_Entregado | 5880 |
| Provision_Constituida | 195 |
| Provision_Automatica | 97 |
| Provision_Manual | 97 |
| Ratio_Provision_Saldo | 0.04 |
| Ratio_Vencido_Saldo | 0.0 |
| Calificacion_Actual | B |
| Estado_Actual | VENCIDO TOTAL |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 50+ |
| Sexo | FEMENINO |
| Estado_Civil | SOLTERO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | FAMILIAR |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | BAJO (<500) |
| Total_Ingresos_Tramo | BAJO (<1000) |
| Patrimonio_Tramo | DESCONOCIDO |
| Es_Dependiente | 0.0 |
| Categoria_Credito | CONSUMO |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **DETERIORA** (41.1 %), no coincide con la etiqueta esperada.

### ds6-B · etiqueta esperada **IGUAL** (datos)

Al día, sin mora y calificación A: en los datos es IGUAL en 100 % (n = 17436).

Valores clave: Estado_Actual = AL DIA, Dias_Morosidad = 0, Calificacion_Actual = A.

Verificación por campo: Estado_Actual: aparece, Dias_Morosidad: aparece, Calificacion_Actual: aparece.

| Campo | Valor |
|---|---|
| Dias_Morosidad | 0 |
| Saldo_Actual | 4271 |
| Saldo_Vencido | 0 |
| Valor_Entregado | 5880 |
| Provision_Constituida | 195 |
| Provision_Automatica | 97 |
| Provision_Manual | 97 |
| Ratio_Provision_Saldo | 0.04 |
| Ratio_Vencido_Saldo | 0.0 |
| Calificacion_Actual | A |
| Estado_Actual | AL DIA |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 50+ |
| Sexo | FEMENINO |
| Estado_Civil | SOLTERO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | FAMILIAR |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | BAJO (<500) |
| Total_Ingresos_Tramo | BAJO (<1000) |
| Patrimonio_Tramo | DESCONOCIDO |
| Es_Dependiente | 0.0 |
| Categoria_Credito | CONSUMO |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **IGUAL** (84.5 %), coincide con la etiqueta esperada.

## ds7_historico_garantias

### ds7-A · etiqueta esperada **SIN_GARANTIA** (datos)

Crédito de CONSUMO en los datos: SIN_GARANTIA en 89,9 % de los casos.

Valores clave: Categoria_Credito = CONSUMO, Para_Vivienda = 0, Valor_Entregado = 1234.

Verificación por campo: Categoria_Credito: aparece, Para_Vivienda: aparece, Valor_Entregado: no aparece.

| Campo | Valor |
|---|---|
| Anio_Desembolso | 2017 |
| Valor_Entregado | 1234 |
| Numero_Cuotas | 30 |
| Tasa_Interes | 14 |
| Para_Vivienda | 0 |
| Es_Agricola_Gan | 0 |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 36-50 |
| Sexo | MASCULINO |
| Estado_Civil | CASADO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | NINGUNA |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | MEDIO (500-1500) |
| Total_Ingresos_Tramo | MEDIO (1000-3000) |
| Patrimonio_Tramo | ALTO (>20000) |
| Es_Dependiente | 0.0 |
| Categoria_Credito | CONSUMO |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **SIN_GARANTIA** (82.8 %), coincide con la etiqueta esperada.

### ds7-B · etiqueta esperada **SIN_GARANTIA** (datos)

Crédito de VIVIENDA en los datos: SIN_GARANTIA en 97,4 % de los casos. Para_Vivienda = 1 no aparece en el dataset, por eso se usa 0.

Valores clave: Categoria_Credito = VIVIENDA, Para_Vivienda = 0, Valor_Entregado = 61250.

Verificación por campo: Categoria_Credito: aparece, Para_Vivienda: aparece, Valor_Entregado: no aparece.

| Campo | Valor |
|---|---|
| Anio_Desembolso | 2017 |
| Valor_Entregado | 61250 |
| Numero_Cuotas | 30 |
| Tasa_Interes | 14 |
| Para_Vivienda | 0 |
| Es_Agricola_Gan | 0 |
| Es_Persona_Juridica | 0 |
| Edad_Tramo | 36-50 |
| Sexo | MASCULINO |
| Estado_Civil | CASADO/A |
| Nivel_Educacion | PRIMARIA |
| Tipo_Vivienda | NINGUNA |
| Cargas_Familiares | 0.0 |
| Ingreso_Mensual_Tramo | MEDIO (500-1500) |
| Total_Ingresos_Tramo | MEDIO (1000-3000) |
| Patrimonio_Tramo | ALTO (>20000) |
| Es_Dependiente | 0.0 |
| Categoria_Credito | VIVIENDA |
| Nombre_Producto | CONSUMO |
| Sucursal | MATRIZ TABACUNDO |
| Canton | PEDRO MONCAYO |

Resultado de referencia de la cadena de inferencia: **COBERTURA_BAJA** (67.2 %), no coincide con la etiqueta esperada.
