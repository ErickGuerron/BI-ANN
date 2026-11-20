# Casos de prueba con valores lejanos al dataset

Dos casos por red para probar si el modelo generaliza:

- **Caso bajo**: numéricos en 0 (o en el mínimo negativo si el campo admite negativos), booleanos en NO, categorías con su primer valor conocido.
- **Caso alto**: numéricos en el máximo de dígitos permitido (por ejemplo 99999 o 9999), booleanos en SI, categorías con su último valor conocido.

Ingréselos tal cual en el asistente. La predicción de cada caso es la que devuelve la cadena de inferencia; compárela con lo que obtenga en la app.

## ds7_historico_garantias

| Campo | Caso bajo | Caso alto |
|---|---|---|
| Anio_Desembolso | 0 | 9999 |
| Valor_Entregado | 0 | 99999 |
| Numero_Cuotas | 0 | 999 |
| Tasa_Interes | 0 | 99 |
| Para_Vivienda | 0 | 1 |
| Es_Agricola_Gan | 0 | 1 |
| Es_Persona_Juridica | 0 | 1 |
| Edad_Tramo | 18-25 | NO APLICA |
| Sexo | FEMENINO | NO APLICA |
| Estado_Civil | CASADO/A | VIUDO/A |
| Nivel_Educacion | BASICA | UNIVERSITARIA |
| Tipo_Vivienda | ARRENDADA | SECUNDARIA |
| Cargas_Familiares | 0.0 | 7.0 |
| Ingreso_Mensual_Tramo | ALTO (>1500) | NO APLICA |
| Total_Ingresos_Tramo | ALTO (>3000) | NO APLICA |
| Patrimonio_Tramo | ALTO (>20000) | ORG MEDIO |
| Es_Dependiente | 0.0 | 1.0 |
| Categoria_Credito | CONSUMO | VIVIENDA |
| Nombre_Producto | CONSUMO | QUIROGRAFARIO VIVIENDA |
| Sucursal | AGENCIA CANGAHUA | OFICINA TABACUNDO |
| Canton | CALDERON | PEDRO MONCAYO |

- Predicción caso bajo: **SIN_GARANTIA** con 85,6 % de confianza; zona de duda: no.
- Predicción caso alto: **SIN_GARANTIA** con 62,0 % de confianza; zona de duda: sí.

## ds1_ingreso_solicitudes

| Campo | Caso bajo | Caso alto |
|---|---|---|
| Anio | 0 | 9999 |
| Total_Ingresos | 0 | 999999 |
| Total_Gastos | 0 | 999999 |
| Monto_Solicitado | 0 | 999999 |
| Mes | 1 | 9 |
| Es_Persona_Juridica | 0 | 1 |
| Edad_Tramo | 18-25 | NO APLICA |
| Sexo | FEMENINO | NO APLICA |
| Estado_Civil | CASADO/A | VIUDO/A |
| Nivel_Educacion | ALFABETIZACION | UNIVERSITARIA |
| Tipo_Vivienda | ARRENDADA | SECUNDARIA |
| Cargas_Familiares | 0.0 | 8.0 |
| Ingreso_Mensual_Tramo | ALTO (>1500) | NO APLICA |
| Total_Ingresos_Tramo | ALTO (>3000) | NO APLICA |
| Patrimonio_Tramo | ALTO (>20000) | ORG MEDIO |
| Es_Dependiente | 0.0 | 1.0 |
| Categoria_Credito | CONSUMO | VIVIENDA |
| Nombre_Producto | CONSUMO | QUIROGRAFARIO VIVIENDA |
| Sucursal | AGENCIA CANGAHUA | OFICINA TABACUNDO |
| Canton | CALDERON | PEDRO MONCAYO |

- Predicción caso bajo: **AJUSTADA** con 57,4 % de confianza; zona de duda: sí.
- Predicción caso alto: **SUPERA** con 66,5 % de confianza; zona de duda: no.

## ds6_gasto_provisiones

| Campo | Caso bajo | Caso alto |
|---|---|---|
| Dias_Morosidad | 0 | 9999 |
| Saldo_Actual | -999999 | 999999 |
| Saldo_Vencido | 0 | 99999 |
| Valor_Entregado | 0 | 99999 |
| Provision_Constituida | 0 | 99999 |
| Provision_Automatica | 0 | 99999 |
| Provision_Manual | 0 | 99999 |
| Ratio_Provision_Saldo | 0.0 | 9.0 |
| Ratio_Vencido_Saldo | 0.0 | 9.0 |
| Calificacion_Actual | A | E |
| Estado_Actual | AL DIA | VENCIDO TOTAL |
| Es_Persona_Juridica | 0 | 1 |
| Edad_Tramo | 18-25 | NO APLICA |
| Sexo | FEMENINO | NO APLICA |
| Estado_Civil | CASADO/A | VIUDO/A |
| Nivel_Educacion | BASICA | UNIVERSITARIA |
| Tipo_Vivienda | ARRENDADA | SECUNDARIA |
| Cargas_Familiares | 0.0 | 7.0 |
| Ingreso_Mensual_Tramo | ALTO (>1500) | NO APLICA |
| Total_Ingresos_Tramo | ALTO (>3000) | NO APLICA |
| Patrimonio_Tramo | ALTO (>20000) | ORG MEDIO |
| Es_Dependiente | 0.0 | 1.0 |
| Categoria_Credito | CONSUMO | VIVIENDA |
| Nombre_Producto | CONSUMO | MICRO SIMPLE TRANSPORTE MOVILIZATE |
| Sucursal | AGENCIA CANGAHUA | OFICINA TABACUNDO |
| Canton | CALDERON | PEDRO MONCAYO |

- Predicción caso bajo: **IGUAL** con 97,6 % de confianza; zona de duda: —.
- Predicción caso alto: **DETERIORA** con 70,8 % de confianza; zona de duda: —.

## ds5_pagos_multimes

| Campo | Caso bajo | Caso alto |
|---|---|---|
| Valor_Entregado | 0 | 99999 |
| Numero_Cuotas | 0 | 999 |
| N_Movimientos | 0 | 9999 |
| N_Cuotas_Con_Pago | 0 | 99 |
| Ultima_Cuota_Pagada | 0 | 99 |
| Monto_Total_Cobrado | 0 | 999999 |
| Monto_Promedio_Mov | 0.0 | 999999.0 |
| Cuotas_Ult_3M | 0 | 99 |
| Cuotas_Ult_6M | 0 | 99 |
| Monto_Ult_3M | 0 | 999999 |
| Monto_Ult_6M | 0 | 999999 |
| Saldo_Ultimo_Pago | 0 | 999999 |
| Dias_Desde_Ultimo_Pago | 0 | 999 |
| Meses_Con_Pagos | 0 | 99 |
| Pct_Cuotas_Pagadas | 0.0 | 9.0 |
| Tasa_Interes | 10.0 | 9.0 |
| Sin_Pagos | 0 | 1 |
| Es_Persona_Juridica | 0 | 1 |
| Edad_Tramo | 18-25 | NO APLICA |
| Sexo | FEMENINO | NO APLICA |
| Estado_Civil | CASADO/A | VIUDO/A |
| Nivel_Educacion | BASICA | UNIVERSITARIA |
| Tipo_Vivienda | ARRENDADA | SECUNDARIA |
| Cargas_Familiares | 0.0 | 7.0 |
| Ingreso_Mensual_Tramo | ALTO (>1500) | NO APLICA |
| Total_Ingresos_Tramo | ALTO (>3000) | NO APLICA |
| Patrimonio_Tramo | ALTO (>20000) | ORG MEDIO |
| Es_Dependiente | 0.0 | 1.0 |
| Categoria_Credito | CONSUMO | VIVIENDA |
| Nombre_Producto | CONSUMO | MICRO SIMPLE TRANSPORTE MOVILIZATE |
| Sucursal | AGENCIA CANGAHUA | OFICINA TABACUNDO |
| Canton | CALDERON | PEDRO MONCAYO |

- Predicción caso bajo: **AL_DIA** con 40,1 % de confianza; zona de duda: sí.
- Predicción caso alto: **MORA_LEVE** con 64,3 % de confianza; zona de duda: no.
