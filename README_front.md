# Front de clasificación · top 4 redes

Cuatro redes (ds7, ds1, ds6 y ds5), ordenadas por AUC de prueba. Cada una tiene su dashboard de resultados y un formulario de ingreso de datos.

## Ejecutar

```
cd "C:\Users\erick\Documents\Sexto Semestre\BI\datasets_multiclase_v4\programa_ann"
pip install -r requirements.txt
streamlit run app.py
```

Si el puerto 8501 está ocupado: `streamlit run app.py --server.port 8502`.

Hace falta la carpeta BI completa, porque la conversión de datos usa `para_companero_cadena_datasets`, `datasets_multiclase_v2/salida` y `datasets_multiclase_v3`.

## Cómo se convierte un dato del formulario

1. Preprocesamiento de `preparacion.py` con `preprocesador.joblib`: limpieza, derivadas, log1p y one-hot.
2. Modelo HistGradientBoosting completo para las variables `GB_*` (solo en ds7, ds1 y ds5).
3. Estandarización con media y desviación de train.
4. NCA de 10 componentes (solo en ds1 y ds6).
5. Red MLP entrenada (`resultados_convergencia/pesos_<ds>.json`).

La cadena se verificó contra las entradas guardadas de la red y las probabilidades de prueba de los cuatro datasets. Coincide con diferencias menores a 5e-7 en las entradas y clase predicha idéntica en 60 filas de prueba y de validación por dataset.

## Validaciones

- Error (bloquea la clasificación): campo vacío, número mal escrito, decimal en un campo entero, valor negativo donde el dataset no tiene negativos.
- Aviso (no bloquea): número fuera del rango observado, o texto que no existe en el dataset (el modelo lo tratará como categoría rara).

No hay combobox, para permitir probar datos nuevos.

## Archivos nuevos

- `app.py`: front nuevo. El anterior quedó en `app_anterior.py`.
- `inferencia.py`: conversión y predicción.
- `construir_auxiliares.py`: genera `auxiliares/<ds>.joblib` (modelo GB, umbrales, estandarización y NCA).
- `generar_campos.py`: genera `auxiliares/campos_<ds>.json` (campos, tipos, rangos y valores conocidos).
- `verificar_inferencia.py`: compara la cadena con los datos guardados.
- `prueba_front.py`: prueba del formulario sin navegador.
- `generar_top3.py` y `top3_<ds>.json`: las 3 mejores configuraciones de cada dataset.
