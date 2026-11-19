# Programa de la ANN (datasets_multiclase_v4)

Red neuronal MLP de scikit-learn para los 8 datasets. Vive aquí para separarla de los datos.

- `ann_core.py`: carga de datos, configuración, entrenamiento con historial, sobremuestreo (solo train) y métricas.
- `ajuste.py`: búsqueda de arquitectura, alpha y sobremuestreo por validación cruzada sobre train. Después reentrena y reporta validación y prueba.
- `resultados/`: JSON con la configuración elegida, la tabla de CV, historial de entrenamiento y CSV de predicciones de prueba.
- `requirements.txt`: dependencias.

Uso:
    pip install -r requirements.txt
    python ajuste.py ds8_notificaciones_cobranza

La prueba (`*_test_nn.csv`) no se usa para elegir nada; solo se reporta.
