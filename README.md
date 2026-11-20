# Programa ANN: clasificador multiclase de cartera

Red neuronal MLP (scikit-learn) para clasificar casos de los 8 datasets de cartera (`datasets_multiclase_v4`). Incluye una app de Streamlit para ingresar un caso y ver la clase predicha con sus probabilidades.

## Estructura

```
programa_ann/
├── app.py                  ← APP: entrada de Streamlit (ejecutar esto)
├── requirements.txt        ← dependencias
├── nucleo/                 ← librerías que se importan (no se ejecutan directo)
│   ├── ann_core.py         carga de datos, configuración de la red, entrenamiento y métricas
│   ├── inferencia.py       convierte un dato del formulario en la entrada de la red y predice
│   ├── frontera.py         dibuja la frontera de decisión en el plano PCA
│   └── visual_red.py       lienzo HTML de las neuronas y conexiones
├── pipeline/               ← scripts que se ejecutan en orden para generar lo que usa la app
│   ├── ajuste_convergencia.py   1. búsqueda de hiperparámetros con parada temprana
│   ├── generar_top3.py          2. resume las 3 mejores configuraciones por dataset
│   ├── construir_auxiliares.py  3. arrays y parámetros que usa la inferencia
│   ├── calibrar.py              3b. temperatura para calibrar la confianza
│   └── generar_campos.py        4. campos del formulario (tipo, rango, valores conocidos)
├── pruebas/                ← verificaciones opcionales; no hacen falta para usar la app
│   ├── verificar_inferencia.py  compara la cadena de inferencia con las salidas guardadas
│   ├── prueba_front.py          prueba la app con casos válidos e inválidos
│   └── depura_front.py          revisa un caso concreto en la app
├── archivo/                ← versiones anteriores; no las usa el front actual
│   ├── app_anterior.py     front previo (8 datasets, entrenamiento en vivo)
│   └── ajuste.py           búsqueda sin parada temprana (escribe en resultados/)
├── docs/                   ← documentación técnica (calibración de probabilidades)
├── .streamlit/config.toml  ← tema de la app
│
│  Salidas generadas (ignoradas por git, se crean al ejecutar el pipeline):
├── resultados_convergencia/   ajuste, pesos, top3 y predicciones por dataset
├── auxiliares/                auxiliares de inferencia y campos del formulario
└── graficos/                  gráficos exportados
```

## Requisitos de carpetas

Las rutas se calculan desde la ubicación del código, así que el repo tiene que estar en este lugar dentro de la carpeta BI:

```
BI/
├── datasets_multiclase_v4/
│   ├── ds1_ingreso_solicitudes/ … ds8_notificaciones_cobranza/   (datos y receta_red_v4.json)
│   └── programa_ann/                                              ← este repo
├── datasets_multiclase_v2/salida/<dataset>/preprocesador.joblib
├── datasets_multiclase_v3/
└── para_companero_cadena_datasets/                                (preparacion.py, preparar_mc.py)
```

Si falta alguna de estas carpetas, la inferencia y el pipeline fallan al importar o al leer archivos. Este repo no incluye los datos.

## Instalación

Python 3.10 o superior.

```bash
cd BI/datasets_multiclase_v4/programa_ann
pip install -r requirements.txt
```

## Proceso de ejecución

Ejecutar desde la carpeta `programa_ann`. Los pasos 1 a 4 se hacen una sola vez por dataset y generan las salidas que la app necesita. Las 4 redes del front son `ds7_historico_garantias`, `ds1_ingreso_solicitudes`, `ds6_gasto_provisiones` y `ds5_pagos_multimes`.

**Paso 1. Ajuste por validación cruzada** (solo con train, con parada temprana)

```bash
python pipeline/ajuste_convergencia.py ds7_historico_garantias
python pipeline/ajuste_convergencia.py ds1_ingreso_solicitudes
python pipeline/ajuste_convergencia.py ds6_gasto_provisiones
python pipeline/ajuste_convergencia.py ds5_pagos_multimes
```

Es el paso más lento. Si se corta por presupuesto de tiempo (`--presupuesto`, en segundos), volver a ejecutar el mismo comando retoma desde el caché. Escribe en `resultados_convergencia/`.

**Paso 2. Top 3 de configuraciones**

```bash
python pipeline/generar_top3.py ds7_historico_garantias ds1_ingreso_solicitudes ds6_gasto_provisiones ds5_pagos_multimes
```

**Paso 3. Auxiliares de inferencia**

```bash
python pipeline/construir_auxiliares.py ds7_historico_garantias ds1_ingreso_solicitudes ds6_gasto_provisiones ds5_pagos_multimes
```

**Paso 3b. Calibración de la confianza** (usa validación; prueba solo se mide)

```bash
python pipeline/calibrar.py ds7_historico_garantias ds1_ingreso_solicitudes ds6_gasto_provisiones ds5_pagos_multimes
```

Guarda `auxiliares/calibracion_<dataset>.json` con la temperatura. Sin ese archivo la app usa T = 1. Proceso y resultados en `docs/calibracion.md`.

**Paso 4. Campos del formulario**

```bash
python pipeline/generar_campos.py
```

Sin argumentos usa las 4 redes del front.

**Paso 5. Abrir la app**

```bash
streamlit run app.py
```

Si el puerto 8501 está ocupado: `streamlit run app.py --server.port 8502`.

## Pruebas (opcional)

Solo hacen falta después de tener los pasos 1 a 5.

```bash
python pruebas/verificar_inferencia.py          # compara la cadena de inferencia con las salidas guardadas
python pruebas/prueba_front.py                  # casos válidos e inválidos en la app
python pruebas/depura_front.py                  # revisa un caso concreto
```

## Cómo funciona la predicción

1. Preprocesamiento de `preparacion.py` con `preprocesador.joblib`: limpieza, variables derivadas, log1p y one-hot.
2. HistGradientBoosting completo para las variables `GB_*` (solo ds7, ds1 y ds5).
3. Estandarización con media y escala de train.
4. NCA de 10 componentes (solo ds1 y ds6).
5. Red MLP entrenada (`resultados_convergencia/pesos_<dataset>.json`): ReLU en las ocultas, softmax en la salida.

La cadena se verificó contra las entradas guardadas de la red y las probabilidades de prueba de los cuatro datasets: diferencias menores a 5e-7 en las entradas y clase predicha idéntica en 60 filas de prueba y de validación por dataset.

## Datasets

| Carpeta | Descripción |
|---|---|
| `ds1_ingreso_solicitudes` | Ingreso de solicitudes (top 4 del front) |
| `ds2_ciclo_solicitud` | Ciclo de solicitud |
| `ds3_desembolsos` | Desembolsos |
| `ds4_cierre_cartera_mensual` | Cierre de cartera mensual |
| `ds5_pagos_multimes` | Pagos multimes (top 4 del front) |
| `ds6_gasto_provisiones` | Gasto de provisiones (top 4 del front) |
| `ds7_historico_garantias` | Histórico de garantías (top 4 del front) |
| `ds8_notificaciones_cobranza` | Notificaciones de cobranza |

## Notas

- La red no supera a la regresión logística en exactitud balanceada en ningún dataset. Sus AUC son comparables a los del árbol de gradiente, así que sirve para ordenar casos. Para decidir la clase conviene la logística o el árbol de v3 (ver `cambios_v4.md`).
- La prueba nunca participa en ninguna decisión; solo se reporta.
