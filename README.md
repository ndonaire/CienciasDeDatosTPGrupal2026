# Productos Bancarios - Propension de Adquisicion

Trabajo integrador de Ciencias de Datos para predecir la probabilidad de que un cliente adquiera un producto de inversion premium.

## Objetivo

Entrenar y validar un modelo con `bank_product_propensity_dataset.csv` y luego aplicar el modelo entrenado sobre `bank_product_propensity_predict.csv` para generar probabilidades de adquisicion por cliente.

## Archivos del proyecto

| Archivo | Descripcion |
|---|---|
| `trabajo_integrador_productos_bancarios.ipynb` | Notebook principal del trabajo, con EDA, ingenieria de variables, validacion del modelo, resultados y recomendaciones. |
| `bank_product_propensity_dataset.csv` | Dataset usado para entrenamiento y validacion. Incluye la variable objetivo `target_product_acquired`. |
| `bank_product_propensity_predict.csv` | Dataset sobre el cual se aplican las predicciones finales. |
| `bank_product_propensity_predictions.csv` | Archivo final generado con las probabilidades estimadas de adquisicion. |
| `model_product_propensity.py` | Script reproducible para entrenar, validar y generar el CSV de predicciones. |
| `model_validation_metrics.json` | Resumen de metricas, distribucion del target, faltantes e importancia de variables. |

## Variable objetivo

La variable objetivo es:

```text
target_product_acquired
```

Donde:

- `0`: el cliente no adquirio el producto.
- `1`: el cliente adquirio el producto.

## Metodologia

El modelo se entreno usando una regresion logistica regularizada. Se eligio este enfoque porque es simple, reproducible e interpretable.

El pipeline incluye:

- Analisis exploratorio breve.
- Revision de valores faltantes.
- Separacion de train/validacion por `customer_id`.
- Imputacion de variables numericas con mediana.
- Indicadores para valores faltantes.
- One-hot encoding para variables categoricas.
- Estandarizacion de variables.
- Ingenieria de variables financieras, transaccionales y de engagement.
- Comparacion entre modelo baseline y modelo final.

## Resultado del modelo

El modelo final obtuvo aproximadamente:

```text
AUC-ROC: 0.805
Accuracy: 0.867
Precision @0.50: 0.665
Recall @0.50: 0.268
F1 @0.50: 0.382
```

La metrica principal es **AUC-ROC**, porque el objetivo del caso es ordenar clientes por propension y priorizar campanias comerciales. El resultado supera el minimo requerido de 0.70 y alcanza el umbral de rendimiento excelente indicado en la consigna.

## Como ejecutar el proyecto

Desde la carpeta del proyecto:

```bash
python model_product_propensity.py
```

Esto vuelve a entrenar el modelo, valida el rendimiento y regenera:

```text
bank_product_propensity_predictions.csv
model_validation_metrics.json
```

Si en Windows `python` no esta configurado, tambien se puede ejecutar desde VS Code o Jupyter abriendo el notebook:

```text
trabajo_integrador_productos_bancarios.ipynb
```

## Archivo de predicciones

El archivo final `bank_product_propensity_predictions.csv` contiene:

| Columna | Descripcion |
|---|---|
| `customer_id` | Identificador del cliente. |
| `snapshot_date` | Fecha del snapshot cliente-mes. |
| `probability_product_acquired` | Probabilidad estimada de adquisicion del producto. |
| `predicted_target_0_50` | Prediccion binaria usando umbral 0.50. |
| `propensity_rank` | Ranking de clientes por probabilidad, donde 1 es la mayor propension. |
| `propensity_decile` | Decil de propension, donde 1 representa el grupo con mayor probabilidad. |

## Interpretacion de negocio

Para una campania comercial, lo mas recomendable es usar `probability_product_acquired`, `propensity_rank` o `propensity_decile`, no solamente la prediccion binaria.

Sugerencia de uso:

- Contactar primero a clientes del decil 1.
- Ampliar a deciles 2 y 3 si el presupuesto de campania lo permite.
- Medir conversion real por decil para recalibrar el modelo en futuras campanias.

## Requisitos

El proyecto usa principalmente:

```text
Python
pandas
numpy
```

No requiere scikit-learn, ya que el modelo esta implementado directamente con `numpy` para facilitar la reproducibilidad.

## Nota

Los datos son sinteticos y corresponden a un escenario ficticio. No contienen informacion real de clientes.
