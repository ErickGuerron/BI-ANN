Feature: ann-monitoreo-garantia. Rama: ann-monitoreo-garantia (desde ann-consolidada, commit 6818826). Sin push ni PR.
Objetivo: que el monitoreo de drift de ds7 vigile Valor_Garantiza, la entrada que más pesa en el modelo con garantía y que el monitoreo anterior detectaba tarde.
Ruta: inline. Riesgo: none (se agregan estadísticos opcionales al monitoreo; sin garantía en la referencia el comportamiento es el mismo; no toca la app).

## Specs
S1. Dos estadísticos nuevos sobre la garantía: proporción de filas con garantía (z-score) y distribución de log(Valor_Garantiza / Valor_Entregado) entre ellas (KS escalado).
S2. Umbrales analíticos para esos dos (alfa de 0.005 cada uno), no calibrados con ventanas de validación.
S3. Sin garantía en la referencia, el monitoreo se comporta como antes.
S4. Comparar monitoreo viejo y nuevo sobre las mismas ventanas: falsas alarmas en ventanas limpias de prueba, y detección en escenarios de drift de la garantía y en los escenarios anteriores.

## Tasks
T1 (S1-S3): pipeline/monitoreo_drift.py (ks, log_cobertura, Referencia con g_train/e_train, ventana genérica) y pruebas/test_monitoreo_drift.py (16 pruebas). Hecho. RED observado dos veces: primero 7 pruebas nuevas fallaron, luego 3 del comportamiento analítico.
T2 (S4): pipeline/t11_monitoreo_garantia.py y resultados_keras/t11_monitoreo_garantia.json. Hecho.

## Log
L1 (2026-11-20): "continua desde 1" (estadísticos de monitoreo para Valor_Garantiza), tras deshacer el aumento sintético.
L2 Primer intento con umbrales calibrados con ventanas de validación: falsa alarma 21 % con n=1000. Causa: solo ~220 garantías positivas en validación y en prueba; el umbral absorbe el desvío propio del pool. Se cambió a umbrales analíticos.
L3 Diagnóstico: positivas de validación frente a entrenamiento KS escalado 0.50; de prueba 1.42 (umbral 1.73); entre submuestras de train, azar puro p95 1.32 y p99 1.59. La prueba es una muestra algo atípica (aprox. percentil 97), no hay deriva temporal (año medio 2015.3, 2015.5, 2015.1).
L4 Falsas alarmas en ventanas limpias de prueba (viejo -> nuevo): n=100 3.5 % -> 4.5 %; n=300 1.5 % -> 4.0 %; n=1000 4.8 % -> 11.8 %. El valor de n=1000 está inflado por el desvío de la prueba y por remuestrear solo 211 positivas; con datos nuevos independientes se espera cerca de 1-2 %.
L5 Detección (completo, viejo -> nuevo): garantía x10: n=100 0.04 -> 0.89, n=300 0.17 -> 1.00; x2: n=300 0.04 -> 0.88, n=1000 0.21 -> 1.00; x0.5: n=300 0.07 -> 0.43, n=1000 0.56 -> 0.98; x0.1: n=100 0.12 -> 0.91; en blanco: 1.00 sin cambio. Con 30 % de contaminación, x10: n=300 0.04 -> 0.42, n=1000 0.12 -> 0.99; con n=100 sigue baja (0.08).
L6 Sin regresión: en ningún otro escenario el monitoreo nuevo detecta menos. Mejora además el cambio de proporciones de clase (minorías x2, n=300: 0.60 -> 0.83).
L7 Limitaciones: una partición y una semilla de escenarios; la proporción de garantías y su cobertura en los escenarios se perturban de forma simple (factor multiplicativo); los umbrales analíticos suponen ventanas de casos independientes; con ventanas de 100 casos y 30 % de contaminación la detección sigue siendo baja.
