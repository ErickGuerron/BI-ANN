Feature: ann-monitoreo-drift. Rama: ann-monitoreo-drift (desde ann-keras-macro-f1, commit a93d733).
Objetivo: probar un monitor de drift a nivel de población, para cubrir lo que el filtro por caso no ve (drift moderado dentro del soporte y cambios en P(y)), y decidir si es viable para el objetivo del proyecto. Sin push ni PR.
Ruta: inline. Riesgo: none (módulo y experimento nuevos, sin tocar código existente ni la app).

## Specs
S1. Comparar una ventana de casos recientes con una referencia mediante cuatro estadísticos: PSI máximo sobre columnas continuas, PSI de la clase predicha, tasa de rechazo OOD y desviación de la tasa de ambigüedad.
S2. Los umbrales se calibran por tamaño de ventana con ventanas remuestreadas de validación (cuantil 0.995), no con 0.1 / 0.25 fijos.
S3. Reportar tasa de falsas alarmas en ventanas limpias de prueba y tasa de detección por tipo de drift, con ventanas de 100, 300 y 1000 casos, completas y con 30 % de contaminación.

## Tasks
T1 (S1, S2): pipeline/monitoreo_drift.py y pruebas/test_monitoreo_drift.py (7 pruebas). Hecho. RED observado: una columna binaria 70/30 producía 3 bordes por la interpolación de np.quantile; se corrigió usando valores observados (method="lower").
T2 (S3): pipeline/t7_monitoreo_drift.py, resultados_keras/t7_resumen.json. Hecho.

## Log
L1 (2026-11-20): "ambas" (cascada en la app y monitoreo de drift, cada una en su rama, para probar y decidir si es viable para el objetivo del proyecto).
L2 Resultados (ds7, 400 ventanas por celda, una partición, una semilla de escenarios):
- Umbrales calibrados (psi_max): n=100 0.699; n=300 0.133; n=1000 0.059. Con ventanas pequeñas el ruido de PSI es grande, por eso no sirven los límites fijos.
- Falsa alarma en ventanas limpias de prueba: n=100 1.8 %; n=300 3.5 %; n=1000 5.8 %. Crece con n porque los umbrales salen de validación y la validación difiere ligeramente de la prueba.
- Drift de covariables en columnas continuas, ventana completa: detección 100 % en todos los tamaños, incluido ruido de 0.5 sd y desplazamiento de 1 sd (el filtro por caso rechazaba 0.8 % y 1.6 %).
- Con 30 % de contaminación: desplazamiento de 1 sd se detecta 15 % con n=100, 100 % con n=300 y 1000; ruido de 0.5 sd 4 %, 63 %, 100 %.
- Cambio de proporciones de clase (minorías x0.5 / x2): ventana completa n=300 33 % / 67 %, n=1000 87 % / 100 %; con 30 % de contaminación solo 7 % / 26 % a n=1000. Es el punto débil.
- Limitaciones: los escenarios perturban las 13 columnas continuas a la vez; el drift de una sola columna no se probó. La referencia son entradas ya estandarizadas, tras el preprocesamiento (que, según la rama ann-app-cascada-ds7, neutraliza valores crudos extremos).
