# ds7: Valor_Garantiza, cascada de decisión y monitoreo de drift

**Rama:** `ann-app-garantia-obligatoria` (publicada en `origin`). Contiene todo lo de este documento.
**Entorno:** el entrenamiento y los experimentos usan el entorno conda `bi` (Python 3.12.3, TensorFlow 2.21.0, Keras 3.15.1, solo CPU). La app y las pruebas de la app corren con NumPy y scikit-learn; no necesitan TensorFlow en ejecución.

## Qué cambió

| Pieza | Qué hace | Archivos |
|---|---|---|
| `Valor_Garantiza` como entrada 97 | La etiqueta es una regla sobre `Valor_Garantiza / Valor_Entregado` (BAJA si es 2 o menos, ALTA si es mayor). La red recibe `log1p(Valor_Garantiza)` estandarizado con la media y la desviación de train. | `nucleo/garantia.py`, `pipeline/t8_valor_garantiza.py` |
| Ensamble de 5 redes Keras | Misma receta que la red de la app: 32 neuronas, ReLU, sobremuestreo, parada temprana. En la app se evalúa con NumPy. | `pipeline/t4_keras.py`, `pipeline/t8_valor_garantiza.py` |
| Cascada por caso | Paso 1: filtro k-NN (k = 5, percentil 99 de validación) rechaza lo que está lejos del entrenamiento. Paso 2: entropía normalizada del ensamble con un solo umbral y piso de cobertura de 0,5 por clase. Salida: `AUTOMATICO` o `REVISION_MANUAL` (`FUERA_DE_RANGO` o `AMBIGUEDAD`). | `nucleo/cascada.py`, `pipeline/politica_abstencion.py`, `pipeline/filtro_ood.py` |
| Monitoreo de drift por ventanas | Compara una ventana de casos con la referencia: PSI máximo en columnas continuas, PSI de la clase predicha, tasa de rechazo, desviación de la ambigüedad y, para la garantía, el z-score de la proporción con garantía y el KS escalado de `log(garantía / monto)`. Solo existe como análisis por lotes; no está conectado a la app. | `pipeline/monitoreo_drift.py`, `pipeline/t7_*.py`, `pipeline/t11_*.py` |
| Formulario de la app | `Valor_Garantiza` es obligatorio. Vacío da error ("Escriba 0 si la operación no tiene garantía"); un 0 escrito es válido y el resultado muestra un aviso. Para ds7 la clase mostrada sale del ensamble. | `app.py`, `pipeline/generar_campos.py` |

## Resultados (conjunto de prueba, ensamble de 5, una partición)

| | Sin garantía | Con `Valor_Garantiza` |
|---|---|---|
| Macro F1 | 0,646 | **0,872** |
| Recall `COBERTURA_BAJA` / `COBERTURA_ALTA` | 0,548 / 0,642 | 0,757 / 0,951 |
| ECE | 0,013 | 0,004 |
| Cascada: cobertura | 85,4 % | 91,4 % |
| Cascada: exactitud de los aceptados | 96,0 % | 99,3 % |
| Cascada: macro F1 contando abstenciones como error | 0,585 | 0,784 |

Con 45 casos externos plausibles (`pruebas/prueba_45_casos_ds7.py`; etiqueta condicional a una garantía supuesta, así que el acuerdo es indicativo): acuerdo 68,9 % → 77,8 %; recall de `COBERTURA_BAJA` 0,20 → 0,40; en los casos que la cascada deja pasar como automáticos, el acuerdo sube de 77,1 % a 92,9 %.

Monitoreo de la garantía (detección en ventanas completas, monitoreo viejo → nuevo): garantía ×10 con 300 casos 0,17 → 1,00; ×2 con 300 casos 0,04 → 0,88; ×0,5 con 1000 casos 0,56 → 0,98. En ningún otro escenario el monitoreo nuevo detecta menos.

## Limitaciones

**Del modelo**
- Todo se midió con **una partición y una semilla de escenarios**; el ensamble se entrenó una vez.
- Como la etiqueta es una regla sobre la garantía y la garantía es entrada, la red aprende en buena parte esa regla. Un vector de ceros con garantía real se acepta al 100 %: las demás variables pesan poco.
- `COBERTURA_BAJA` sigue débil donde faltan datos: con cobertura menor a 1 casi no hay ejemplos y la masa de BAJA está entre 1,4 y 1,9, pegada al umbral de 2. En los 45 casos acierta 1 de 4 con cobertura 0,55 y 1 de 5 entre 1,75 y 1,95.
- El 49 % de los `COBERTURA_BAJA` de entrenamiento (377 de 769) tienen garantía en 0 con `Tiene_Garantia = 1`; por montos se confunden con `SIN_GARANTIA`. Falta aclarar si es "sin dato" o "garantía real en 0" (pendiente 6 de `pendientes.md`).
- Se probó y se **descartó** un aumento con filas sintéticas (garantía = k × monto, etiquetadas por la regla): +0,007 de macro F1 en validación, recall de ALTA −0,016 y pureza cruzada de ALTA −0,024 (supera el límite de 0,02). Se deshizo; no hay datos sintéticos en uso.

**De la cascada**
- Un campo crudo extremo aislado no se detecta: el preprocesamiento lo neutraliza (`Anio_Desembolso = 9999` sube la aceptación a 92,5 %).
- No ve cambios en la garantía: con garantía ×10 sigue aceptando el 94 % de los casos.

**Del monitoreo**
- Necesita ventanas de al menos 300 casos; con 100 casos y 30 % de contaminación detecta solo el 8 %.
- Falsas alarmas en ventanas limpias de prueba: 4,5 % (100 casos), 4,0 % (300) y 11,8 % (1000). El valor de 1000 está inflado porque la prueba es una muestra algo atípica (solo 211 garantías positivas, KS escalado 1,42 frente a un percentil 95–99 de 1,32–1,59 por azar); con datos nuevos se espera cerca de 1–2 %, sin comprobar.
- Los umbrales de los dos estadísticos de garantía son analíticos y suponen ventanas de casos independientes.
- Los escenarios de drift de la garantía son factores multiplicativos simples.

**De la app**
- El asistente del formulario (`st.dialog`) no se ejercitó de punta a punta; se probaron la validación, los textos y el arranque.
- Un 0 escrito por desconocimiento sigue siendo posible; el aviso lo hace visible pero no lo impide.
- El monitoreo no está conectado: el historial de la app vive en la sesión de Streamlit y se pierde al cerrar.

## Cómo reproducirlo

Los artefactos de `auxiliares/`, `resultados_convergencia/` y los modelos Keras (`resultados_keras/t5_miembros/`, `t8_miembros/`) no se versionan; se regeneran en este orden, con `conda run -n bi python`:

1. `pipeline/t5_abstencion_por_clase.py`: entrena y guarda los 5 miembros sin garantía.
2. `pipeline/exportar_cascada_ds7.py`: exporta la cascada sin garantía.
3. `pipeline/t8_valor_garantiza.py`: entrena los 5 miembros con garantía, escribe `garantia_ds7_historico_garantias.json` y la cascada con sufijo `_garantia`.
4. `python pipeline/generar_campos.py ds7_historico_garantias` (Python base): regenera los campos del formulario, con `Valor_Garantiza`.
5. Comparaciones: `pipeline/t9_comparacion.py`, `pipeline/t11_monitoreo_garantia.py`; prueba de 45 casos: `python pruebas/prueba_45_casos_ds7.py [ruta_del_csv]`.

Pruebas: `python -m unittest pruebas.test_filtro_ood pruebas.test_politica_abstencion pruebas.test_monitoreo_drift pruebas.test_garantia pruebas.test_cascada` (49) en el entorno `bi`, y `python -m unittest pruebas.test_app_campo_opcional` (13) con el Python base.

## Siguiente paso sugerido

Una pestaña "Monitoreo" en la app donde se **sube un CSV de operaciones recientes** y se muestran los estadísticos y las alertas. Da el beneficio del monitoreo bajo demanda sin guardar datos personales ni agregar persistencia. El monitoreo continuo conviene dejarlo para cuando haya tráfico real: con el uso actual (pocas operaciones por sesión) casi nunca se alcanzarían los 300 casos de una ventana.
