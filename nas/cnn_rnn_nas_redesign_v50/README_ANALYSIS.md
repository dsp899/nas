# README_ANALYSIS

Este documento describe **cómo interpretar** la carpeta `analysis/` generada por:

```bash
python3 run.py nas plot --summary-json <ruta_a_search_summary.json>
```

La intención es que `analysis/` sea autosuficiente para leer una búsqueda NAS ya terminada sin tener que volver al código.

---

## 1. Artefactos fuente del análisis

El análisis siempre parte de estos tres artefactos del run NAS:

- `search_summary.json`
- `search_architectures.csv`
- `search_controller_history.csv`

### `search_summary.json`
Se usa para:
- resolver rutas a los artefactos del run
- leer metadatos del experimento
- recuperar el `search_space`
- leer parámetros globales del run como `rolling_window`
- recuperar resúmenes por epoch registrados durante la búsqueda

### `search_architectures.csv`
Es la fuente principal para el análisis de candidatas. Se usa para:
- accuracies de cada arquitectura sampleada
- orden global de sampleo
- epoch de búsqueda en el que apareció cada candidata
- campos del search space (cnn, rnn, layers, units, direction, memory_mode, seq, etc.)
- indicador `cached`

A partir de este CSV se calculan:
- métricas por muestra
- métricas por epoch
- importance por dimensión
- interacciones por pares
- profiles por dimensión
- correlaciones por valor

### `search_controller_history.csv`
Se usa para:
- `loss` del controller
- `learning_rate` del controller
- evolución interna del entrenamiento del controller por epoch de búsqueda

---

## 2. Estructura de la carpeta `analysis/`

La estructura actual es:

```text
analysis/
  analysis_manifest.json

  tables/
    metrics_by_sample.csv
    metrics_by_epoch.csv
    dimension_value_stats.csv
    dimension_importance.csv
    pairwise_interactions.csv

  overview/
    epoch_accuracy_cumulative.png
    epoch_accuracy_rolling.png
    controller_loss.png
    layer_distribution_by_epoch.png
    layer_distribution_cumulative.png

  search_space/
    dimension_importance.png
    pairwise_interactions.png

  dimensions/
    <dimension>/
      cumulative_mean_accuracy_vs_cumulative_selection_count.png
      cumulative_mean_accuracy_vs_cumulative_selection_share.png
      cumulative_median_accuracy_vs_cumulative_selection_count.png
      cumulative_median_accuracy_vs_cumulative_selection_share.png
      value_trajectory_summary.csv

  correlations/
    dimension_selection_accuracy_alignment.csv
    dimension_selection_accuracy_alignment.png

    by_value_cumulative/
      <dimension>_value_correlations_cumulative.csv
      <dimension>_value_correlations_cumulative.png

    by_value_local/
      <dimension>_value_correlations_local.csv
      <dimension>_value_correlations_local.png

    by_dimension_over_time/
      <metric>.csv
      <metric>.png
```

---

## 3. Subdirectorio `tables/`

### `metrics_by_sample.csv`
Una fila por arquitectura sampleada.

Fuente:
- `search_architectures.csv`

Incluye:
- accuracy y métricas relacionadas
- orden de sampleo global
- epoch y orden dentro del epoch
- campos del search space
- información de cacheo
- métricas rolling/cumulative a nivel de muestra si el módulo las construye para apoyo interno

Uso principal:
- trazabilidad de candidatas
- análisis fino por muestra
- base para los profiles y las estadísticas por dimensión

---

### `metrics_by_epoch.csv`
Una fila por epoch de búsqueda.

Fuente:
- agregación de `metrics_by_sample.csv`
- información complementaria de `search_summary.json`
- información complementaria de `search_controller_history.csv`

Incluye típicamente:
- `epoch_mean_accuracy`
- `epoch_median_accuracy`
- `epoch_max_accuracy`
- `epoch_min_accuracy`
- métricas rolling por epoch
- métricas cumulative por epoch
- distribución de `layers` por epoch
- loss final del controller en ese epoch
- intentos de sampling, duplicados y fallback si existen en el run

Uso principal:
- gráficas de evolución global del search
- análisis de convergencia

---

### `dimension_value_stats.csv`
Una fila por **valor** dentro de una **dimensión**.

Fuente:
- agrupación de `metrics_by_sample.csv` por `(dimension, value)`

Incluye:
- `count`
- `sample_fraction`
- `cached_fraction`
- `mean_accuracy`
- `median_accuracy`
- `std_accuracy`
- `max_accuracy`
- `min_accuracy`
- `best_signature`

Uso principal:
- ordenar valores dentro de una dimensión
- construir perfiles por dimensión
- saber qué valores son más frecuentes o más fuertes

---

### `dimension_importance.csv`
Una fila por dimensión.

Fuente:
- `metrics_by_sample.csv`
- `search_space` del `summary`

Incluye:
- `configured_option_count`
- `sampled_unique_values`
- `sample_coverage`
- `sampling_entropy`
- `eta_squared_accuracy`
- `eta_squared_search_accuracy`
- `mean_accuracy_range`
- `max_accuracy_range`
- `best_value_by_mean`
- `best_value_by_max`
- `relative_importance`

#### Interpretación
- `eta_squared_accuracy`: fracción de varianza de `accuracy` explicada por esa dimensión en las arquitecturas sampleadas
- `relative_importance`: normalización de la importancia relativa entre dimensiones analizadas
- `sample_coverage`: qué parte del vocabulario configurado de esa dimensión ha aparecido realmente en el muestreo
- `sampling_entropy`: si el muestreo está muy concentrado en pocos valores o repartido entre varios

---

### `pairwise_interactions.csv`
Una fila por par de dimensiones.

Fuente:
- `metrics_by_sample.csv`
- pares de columnas del search space

Incluye medidas tipo:
- `eta_squared_accuracy`
- rangos de accuracy por combinación
- cobertura de combinaciones sampleadas

Uso principal:
- detectar pares de dimensiones que parecen interactuar de manera importante en la accuracy

---

## 4. Subdirectorio `overview/`

Contiene gráficas generales de la búsqueda.

### `epoch_accuracy_cumulative.png`
Representa métricas acumuladas por epoch.

Fuente:
- `metrics_by_epoch.csv`

Incluye curvas como:
- `cumulative_mean_accuracy`
- `cumulative_median_accuracy`
- `cumulative_max_accuracy`
- `cumulative_min_accuracy`
- y, si aplica, `cumulative_best_*`

#### Interpretación
Estas curvas responden a cómo evoluciona el comportamiento agregado de la búsqueda a medida que se acumulan epochs.

---

### `epoch_accuracy_rolling.png`
Representa métricas rolling por epoch.

Fuente:
- `metrics_by_epoch.csv`
- ventana `rolling_window` tomada del `summary`

Incluye curvas como:
- `rolling_mean_accuracy`
- `rolling_median_accuracy`
- `rolling_max_accuracy`
- `rolling_min_accuracy`
- y, si aplica, `rolling_best_*`

#### Interpretación
Es la gráfica más útil para ver tendencia local y no solo acumulación histórica.

---

### `controller_loss.png`
Representa la pérdida del controller.

Fuente:
- `search_controller_history.csv`

Incluye normalmente:
- loss por step del controller
- resumen por epoch si el plotting lo calcula

#### Interpretación
Sirve para diagnosticar el comportamiento del controller, no para medir directamente calidad del search.

---

### `layer_distribution_by_epoch.png`
Distribución de `layers` por epoch.

Fuente:
- `metrics_by_epoch.csv`
- datos registrados en `search_summary.json`

#### Interpretación
Ayuda a ver si el sampler se colapsa hacia arquitecturas de 1, 2 o 3 capas.

---

### `layer_distribution_cumulative.png`
Distribución acumulada de `layers`.

Fuente:
- `metrics_by_epoch.csv`

#### Interpretación
Muestra cómo queda el historial acumulado del muestreo por número de capas.

---

## 5. Subdirectorio `search_space/`

### `dimension_importance.png`
Versión gráfica de `dimension_importance.csv`.

Fuente:
- `dimension_importance.csv`

#### Interpretación
Permite identificar rápidamente qué dimensiones tienen mayor capacidad explicativa sobre la accuracy observada en las candidatas sampleadas.

---

### `pairwise_interactions.png`
Versión gráfica de `pairwise_interactions.csv`.

Fuente:
- `pairwise_interactions.csv`

#### Interpretación
Ayuda a detectar qué pares de dimensiones parecen relacionarse con cambios importantes en el rendimiento.

---

## 6. Subdirectorios `dimensions/<dimension>/`

Cada dimensión tiene su propio subdirectorio.

Ejemplo:

```text
analysis/dimensions/layers/
analysis/dimensions/rnn/
analysis/dimensions/memory_mode/
```

Cada uno contiene:

### Las 4 gráficas acumuladas por dimensión
Se cruzan dos tipos de accuracy con dos tipos de selección:

#### Accuracy acumulada
- `cumulative_mean_accuracy`
- `cumulative_median_accuracy`

#### Selección acumulada
- `cumulative_selection_count`
- `cumulative_selection_share`

Eso produce cuatro combinaciones:
- `cumulative_mean_accuracy_vs_cumulative_selection_count.png`
- `cumulative_mean_accuracy_vs_cumulative_selection_share.png`
- `cumulative_median_accuracy_vs_cumulative_selection_count.png`
- `cumulative_median_accuracy_vs_cumulative_selection_share.png`

#### Fuente de cálculo
Estas gráficas se construyen a partir de un **profile por valor y por epoch** derivado desde `metrics_by_sample.csv`:
- se agrupa por `(search_epoch, value)`
- se calcula selección por epoch
- se construyen series acumuladas por valor a lo largo del tiempo

#### Interpretación
Sirven para ver, para cada valor de la dimensión:
- cómo se consolida su accuracy acumulada
- cómo gana o pierde presencia en el muestreo

---

### `value_trajectory_summary.csv`
Resumen por valor dentro de la dimensión.

Fuente:
- profile temporal por valor construido desde `metrics_by_sample.csv`

Incluye típicamente:
- `first_epoch_seen`
- `last_epoch_seen`
- `final_cumulative_mean_accuracy`
- `final_cumulative_median_accuracy`
- `final_selection_count`
- `final_selection_share`
- pendientes (`*_slope`)
- correlaciones acumuladas y/o locales según la versión actual del análisis

#### Interpretación
Es el mejor resumen tabular para entender qué valores de una dimensión:
- terminan dominando
- terminan rindiendo mejor
- mejoran o empeoran con el tiempo

---

## 7. Subdirectorio `correlations/`

Esta sección está dedicada específicamente a la relación entre **selección** y **rendimiento**.

### 7.1 `dimension_selection_accuracy_alignment.csv`
Resumen global por dimensión.

### 7.2 `dimension_selection_accuracy_alignment.png`
Versión gráfica del CSV anterior.

#### Qué visualiza
Para cada dimensión se calcula una correlación entre:
- `final_selection_share` de cada valor de la dimensión
- `final_cumulative_median_accuracy` o `final_cumulative_mean_accuracy` de esos mismos valores

Es decir, dentro de una dimensión:
- cada **valor** es un punto
- X = cuota final de selección
- Y = rendimiento final acumulado

Luego se resume eso como una correlación por dimensión.

#### Interpretación
Responde a:
> ¿El sampler acaba favoreciendo los valores que mejor rendimiento final muestran dentro de esta dimensión?

- positiva alta → buena alineación entre selección y rendimiento
- cerca de 0 → alineación débil
- negativa → el sampler tiende a favorecer valores que no son los más rentables

---

### 7.3 `by_value_cumulative/`
Contiene correlaciones **por valor** usando series **acumuladas**.

Por cada dimensión:
- `<dimension>_value_correlations_cumulative.csv`
- `<dimension>_value_correlations_cumulative.png`

#### Fuente
Se parte del profile acumulado por valor y por epoch de esa dimensión.

#### Qué se correlaciona
Para cada valor, a lo largo de los epochs válidos:
- `cumulative_selection_count` vs `cumulative_mean_accuracy`
- `cumulative_selection_count` vs `cumulative_median_accuracy`
- `cumulative_selection_share` vs `cumulative_mean_accuracy`
- `cumulative_selection_share` vs `cumulative_median_accuracy`

#### Interpretación
Describen si, para un valor dado:
- conforme gana historia y presencia acumulada,
- también mejora o empeora su rendimiento acumulado

Estas correlaciones son más suaves y describen consolidación histórica.

---

### 7.4 `by_value_local/`
Contiene correlaciones **por valor** usando métricas **locales por epoch**.

Por cada dimensión:
- `<dimension>_value_correlations_local.csv`
- `<dimension>_value_correlations_local.png`

#### Fuente
Se parte de las métricas locales por epoch para cada valor:
- `epoch_selection_count`
- `epoch_selection_share`
- `epoch_mean_accuracy`
- `epoch_median_accuracy`

#### Qué se correlaciona
Para cada valor, a lo largo de los epochs válidos:
- `epoch_selection_count` vs `epoch_mean_accuracy`
- `epoch_selection_count` vs `epoch_median_accuracy`
- `epoch_selection_share` vs `epoch_mean_accuracy`
- `epoch_selection_share` vs `epoch_median_accuracy`

#### Restricción importante
Estas correlaciones solo se calculan si el valor aparece en al menos **N epochs válidos**. Ese umbral se usa para evitar correlaciones espurias con soporte ridículo.

#### Interpretación
Son más adecuadas para responder:
> en los epochs donde este valor se selecciona más, ¿también rinde mejor?

Estas correlaciones son más sensibles al comportamiento local del sampler.

---

### 7.5 `by_dimension_over_time/`
Contiene correlaciones por **dimensión y por epoch**, separadas por variante.

La idea es ver cómo evoluciona en el tiempo la alineación entre selección y rendimiento dentro de cada dimensión.

Típicamente se generan CSV/figuras para variantes como:
- local + median
- local + mean
- cumulative + median
- cumulative + mean

#### Qué responde
No solo si una dimensión acaba alineada al final, sino:
- si se alinea pronto
- si mejora progresivamente
- si oscila
- si colapsa

Esta es la lectura temporal principal del bloque de correlaciones.

---

## 8. `analysis_manifest.json`

Es el manifiesto general del análisis.

Contiene:
- artefactos fuente utilizados
- rutas de salida generadas
- metadatos del search space
- semántica de algunas métricas

Uso principal:
- trazabilidad
- saber exactamente qué análisis se generó y a partir de qué run

---

## 9. Resumen rápido de interpretación

Si quieres una lectura rápida de `analysis/`, el orden recomendado es:

1. `overview/epoch_accuracy_rolling.png`  
   → tendencia local real del rendimiento de la búsqueda

2. `overview/layer_distribution_by_epoch.png`  
   → comprobar si el sampler colapsa estructuralmente

3. `search_space/dimension_importance.png`  
   → qué dimensiones parecen importar más

4. `correlations/dimension_selection_accuracy_alignment.png`  
   → si el sampler termina favoreciendo valores buenos por dimensión

5. `correlations/by_dimension_over_time/`  
   → cómo evoluciona esa alineación en el tiempo

6. `dimensions/<dimension>/...` y `value_trajectory_summary.csv`  
   → análisis detallado de una dimensión concreta

7. `correlations/by_value_cumulative/` y `correlations/by_value_local/`  
   → análisis fino por valor dentro de una dimensión

---

## 10. Idea clave

La carpeta `analysis/` no es solo un conjunto de gráficos sueltos. Está organizada para responder tres preguntas:

1. **¿Cómo evoluciona la búsqueda en global?**  
   → `overview/`

2. **¿Qué partes del search space parecen importar más?**  
   → `search_space/`

3. **¿Cómo se relacionan selección y rendimiento, tanto por dimensión como por valor?**  
   → `dimensions/` y `correlations/`

Si quieres, en una siguiente versión puedo hacer que este README se copie automáticamente dentro de cada carpeta `analysis/` generada por `run.py nas plot`, para que el análisis quede autocontenido junto a sus gráficos y tablas.
