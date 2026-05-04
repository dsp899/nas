# rnn_benchlib

`rnn_benchlib` genera, benchmarkea y reutiliza modelos RNN para clasificación de vídeo, y además exporta datasets para el módulo `gnn_latency`.

## Principios del diseño actual

- un **modelo** es global y reusable en `artifacts/models/<model_id>/`
- un **lote** queda definido solo por:
  - `search_space`
  - `generation_seed`
  - `requested_count`
- el lote se materializa en `artifacts/lots/<lot_id>/`
- una nueva ejecución del mismo lote **reescribe la vista del lote**, pero **reutiliza** los modelos ya persistidos
- la configuración operativa se carga desde un fichero centralizado en `./configs/`
- el CLI usa siempre `./artifacts` como raíz de persistencia

## Dónde poner la configuración

Usa un directorio común a nivel de repositorio:

```text
configs/
  rnn_config_example.json
  rnn_ucf101_seed1234.json
  rnn_benchmark_fenix_small.json
```

La convención recomendada es que el nombre del fichero deje claro que pertenece a la rama RNN (`rnn_*.json`).

Hay un ejemplo base en:

```text
configs/rnn_config_example.json
```

## Estructura de persistencia

### Modelos globales

```text
artifacts/models/<model_id>/
  meta/
    spec.json
    manifest.json
  graphs/
    graph_record.json
    encoder_tflite_graph.json
    head_tflite_graph.json
  benchmarks/<profile_id>/
    float.jsonl
    float_measurements.jsonl
    tflite.jsonl
    tflite_measurements.jsonl
```

### Lotes

```text
artifacts/lots/<lot_id>/
  lot.json
  members.jsonl
  generation/
    config.json
    summary.json
    logs/latest/
      runtime.json
      status.json
      progress.jsonl
      errors.jsonl
  benchmarks/<benchmark_id>/
    config.json
    summary.json
    profiles/<profile_id>.json
    logs/latest/
      runtime.json
      status.json
      progress.jsonl
      errors.jsonl
  gnn_latency/datasets/<dataset_id>/
    export.json
    rows.jsonl
    splits/
    runs/
```

## CLI mínima

### Generar o recalcular un lote

```bash
python -m rnn_benchlib.cli.run --op generate --config ./configs/rnn_config_example.json
```

### Benchmarkear un lote

```bash
python -m rnn_benchlib.cli.run --op benchmark --config ./configs/rnn_config_example.json
```

### Exportar dataset para la GNN

```bash
python -m rnn_benchlib.cli.run --op export-dataset --config ./configs/rnn_config_example.json
```

### Crear split train/val/test

```bash
python -m rnn_benchlib.cli.run --op make-split --config ./configs/rnn_config_example.json
```

### Entrenar la GNN

```bash
python -m rnn_benchlib.cli.run --op train-gnn --config ./configs/rnn_config_example.json
```

Al terminar el entrenamiento, el run exporta automáticamente:

- `metrics.json`: resumen del entrenamiento y métricas finales
- `history.json`: historial por época
- `figures/loss.png`: curva de `train_loss`
- `figures/wall_mape.png`: curvas de `train_wall_mape` y `val_wall_mape`

Si en el futuro el historial incorpora más métricas `train_*`/`val_*`, también se exportará una figura adicional por métrica dentro de `figures/`.

### Evaluar la GNN

```bash
python -m rnn_benchlib.cli.run --op eval-gnn --config ./configs/rnn_config_example.json
```

## Guía detallada del fichero de configuración

El fichero de configuración centraliza toda la configuración de la rama RNN. La idea es separar claramente:

- la **identidad lógica** del lote y del benchmark
- la **configuración operativa** de generación y benchmark en paralelo
- la **configuración derivada** para exportar dataset, crear splits y entrenar la GNN

Los bloques esperados son:

- `LOT`
- `GENERATION.runtime`
- `BENCHMARK`
- `DATASET`
- `SPLIT`
- `GNN`

A continuación se explica cada bloque y todos sus campos del ejemplo `configs/rnn_config_example.json`.

### 1. `LOT`

Define la identidad lógica del lote. Dos ejecuciones producen el **mismo lote** si y solo si coinciden estos tres elementos:

- `search_space`
- `generation_seed`
- `requested_count`

#### `LOT.search_space`

Es el espacio de búsqueda de modelos RNN. Cada clave representa una dimensión del muestreo.

##### `layers`
Número de capas recurrentes permitidas.

- ejemplo: `[1, 2, 3]`
- uso: define si los modelos del lote pueden tener una, dos o tres capas RNN

##### `rnn`
Tipos de celda recurrente permitidos.

- valores típicos: `lstm`, `gru`
- uso: define qué familias de RNN pueden aparecer en el lote

##### `units_0`, `units_1`, `units_2`
Número de unidades permitido por capa.

- `units_0` corresponde a la primera capa
- `units_1` corresponde a la segunda capa
- `units_2` corresponde a la tercera capa
- el valor `0` en capas superiores se usa como forma explícita de “capa no usada” cuando el modelo tiene menos capas que el máximo soportado

##### `direction`
Direccionalidad permitida para la RNN.

- valores típicos: `unidirectional`, `bidirectional`
- uso: controla si una capa procesa la secuencia en un solo sentido o en ambos

##### `memory_mode`
Modo de memoria temporal entre clips o pasos.

- valores típicos: `none`, `carry_forward`
- uso: controla si el estado de la RNN se propaga o no entre clips

##### `seq`
Longitudes de secuencia permitidas para la entrada del modelo.

- ejemplo: `[3, 6, 9, 12]`
- uso: define cuántos pasos temporales procesa cada clip

##### `head_units`
Tamaños permitidos para el bloque denso final o “head”.

- ejemplo: `[128, 256, 512, 1024]`
- uso: controla la capacidad del clasificador final sobre la representación del encoder

##### `video_decision`
Estrategia de agregación a nivel de vídeo.

- ejemplos: `average`, `majority`, `max_prob`
- uso: define cómo combinar predicciones o puntuaciones de clips para producir una decisión final por vídeo

##### `video_decision_input`
Tipo de señal usada por la agregación a nivel de vídeo.

- ejemplos: `clip_logits`, `clip_embeddings`
- uso: especifica si la decisión de vídeo se toma a partir de logits por clip o embeddings por clip

#### `LOT.generation_seed`
Semilla del proceso de muestreo.

- uso: garantiza reproducibilidad del lote
- efecto: cambiarla cambia el lote, aunque el `search_space` sea el mismo

#### `LOT.requested_count`
Número de modelos pedidos para el lote.

- uso: controla cuántos miembros debe contener el lote final
- efecto: cambiarlo cambia el lote

#### `LOT.experiment`
Metadatos experimentales asociados al lote. En el ejemplo actual no forman parte de la firma del lote; sirven para contextualizar cómo se interpretan las redes y los datos.

##### `dataset_profile`
Nombre corto del escenario o dataset al que se asocia el experimento.

- ejemplo: `ucf101`
- uso: ayuda a etiquetar el contexto del lote

##### `num_classes`
Número de clases de salida del modelo.

- uso: controla la dimensionalidad de la capa final de clasificación

##### `feature_dim`
Dimensión de la característica por paso temporal o por clip.

- uso: define el tamaño del vector de entrada que consume la RNN

##### `video_steps`
Número de pasos o clips por vídeo en el experimento.

- uso: documenta el tamaño temporal global del ejemplo de vídeo

### 2. `GENERATION.runtime`

Contiene únicamente configuración operativa de la generación paralela. No cambia la identidad del lote.

#### `jobs`
Número de workers paralelos o modo automático.

- valor típico: `auto`
- uso: deja al scheduler decidir cuántos procesos lanzar según CPU y memoria disponibles

#### `jobs`
Estimación conservadora de memoria por worker de generación.

- uso: el scheduler la usa para decidir cuántos workers puede despachar sin sobrepasar RAM
- recomendación: calibrarla con un script de medición realista de generación/conversión

#### `max_ram_fraction`
Fracción máxima de la RAM total que la generación puede ocupar.

- ejemplo: `0.65`
- uso: impone un techo relativo de uso de memoria

#### `ram_reserve_mb`
Colchón absoluto de RAM libre que debe quedar disponible.

- ejemplo: `32768`
- uso: evita apurar demasiado la memoria, incluso si el porcentaje todavía permitiría lanzar más workers

#### `worker_max_tasks`
Número máximo de tareas que puede ejecutar un worker antes de reciclarse.

- uso: reduce acumulación residual de memoria entre modelos
- cuanto menor, más limpio pero con mayor sobrecoste por recreación de proceso

#### `intra_op_threads`
Número de hilos intra-op de TensorFlow por worker.

- uso: controla el paralelismo interno de kernels y operaciones individuales

#### `inter_op_threads`
Número de hilos inter-op de TensorFlow por worker.

- uso: controla el paralelismo entre operaciones independientes

### 3. `BENCHMARK`

Se divide en dos subbloques:

- `signature`: define el benchmark lógico reusable
- `runtime`: define cómo se ejecuta ese benchmark en paralelo

#### 3.1 `BENCHMARK.signature`

Estos campos sí forman la identidad lógica del benchmark sobre un lote.

##### `feature_source`
Origen de las features usadas durante el benchmark.

- ejemplo: `synthetic`
- uso: distingue si las entradas vienen de generación sintética, de un fichero o de otro backend

##### `num_videos`
Número de vídeos o secuencias a procesar por modelo durante el benchmark.

- uso: controla el volumen de evaluación por modelo

##### `feature_seed`
Semilla para la generación o muestreo de features de benchmark.

- uso: hace reproducibles las entradas sintéticas del benchmark

##### `distribution`
Distribución usada para generar las features sintéticas.

- ejemplo: `normal`
- uso: controla la estadística de las entradas si `feature_source` es sintético

##### `warmup_runs`
Número de ejecuciones de calentamiento antes de medir.

- uso: estabiliza caches, inicialización y optimizaciones antes de registrar métricas

##### `steady_runs`
Número de ejecuciones medidas que se usarán para construir las métricas finales.

- uso: cuanto mayor, más estable suele ser la media, a costa de tiempo total

##### `threads`
Número de hilos usados por el backend de benchmark para cada medición.

- uso: forma parte de la medición lógica porque puede cambiar latencias y throughput observados

##### `hardware_target`
Nombre lógico del host o target de benchmark.

- ejemplo: `fenix_cpu`
- uso: separa campañas de benchmark realizadas en entornos distintos

##### `experiment_name`
Nombre humano de la campaña de benchmark.

- uso: ayuda a identificarla en la persistencia y en informes
- en el estado actual conviene mantenerlo estable si quieres reutilizar exactamente el mismo benchmark lógico

#### 3.2 `BENCHMARK.runtime`

Estos campos son operativos y no deben cambiar el `benchmark_id`.

##### `jobs`
Número de workers paralelos o modo automático para benchmark.

##### `cpu_policy`
Política de afinidad o reparto de CPU entre workers.

- ejemplo: `none`
- uso: define si el scheduler intenta fijar afinidad o deja al SO la gestión

##### `cpu_reserve_cores`
Número de cores a dejar sin ocupar por el benchmark.

- uso: deja margen para el sistema y otros procesos

##### `jobs`
Estimación de RAM por worker de benchmark.

- uso: análogo al bloque de generación, pero calibrado para la fase de benchmark

##### `max_ram_fraction`
Techo relativo de RAM para benchmark.

##### `ram_reserve_mb`
Colchón absoluto de RAM libre para benchmark.

##### `worker_max_tasks`
Máximo de tareas por worker antes de reciclar el proceso.

##### `runtime`
Qué perfiles materializar al benchmarkear.

- valores típicos: `float`, `tflite`, `both`
- uso: decide si medir solo el perfil float, solo el perfil TFLite o ambos
- nota importante: este campo **no** cambia el `benchmark_id`; solo controla qué perfiles se generan dentro del mismo benchmark lógico

### 4. `DATASET`

Controla cómo se exporta el dataset para `gnn_latency` a partir de un lote ya benchmarkeado.

#### `runtime_scope`
Filtra qué perfiles del benchmark incluir en el dataset exportado.

- valores típicos: `float`, `tflite`, `both`
- uso:
  - `float` exporta solo muestras del perfil float
  - `tflite` exporta solo muestras del perfil TFLite
  - `both` exporta ambas
- nota importante: este campo no cambia el `benchmark_id`; solo filtra perfiles ya existentes

### 5. `SPLIT`

Controla la partición train/val/test del dataset exportado.

#### `train_ratio`
Fracción del dataset asignada a entrenamiento.

#### `val_ratio`
Fracción del dataset asignada a validación.

#### `test_ratio`
Fracción del dataset asignada a test.

- recomendación: las tres proporciones deben sumar 1.0

#### `seed`
Semilla de la partición.

- uso: hace reproducible el split

### 6. `GNN`

La configuración de la GNN se organiza sin redundancia en cuatro subbloques:

- `GNN.run_name`: nombre lógico del run de entrenamiento/evaluación
- `GNN.seed`: semilla del entrenamiento
- `GNN.model`: arquitectura del predictor
- `GNN.training`: hiperparámetros del loop de entrenamiento
- `GNN.optimizer`: configuración del optimizador
- `GNN.runtime`: configuración de dispositivo y visibilidad del progreso

#### `GNN.run_name`
Nombre del run de entrenamiento. Cambiarlo genera un `run_id` distinto dentro del mismo dataset/split.

#### `GNN.seed`
Semilla del entrenamiento. Controla el barajado y la inicialización reproducible del trainer.

#### `GNN.model`
Define la arquitectura del predictor GNN.

- `hidden_dim`: dimensión oculta de nodos `op` y `tensor`
- `graph_hidden_dim`: dimensión del MLP de contexto global
- `num_layers`: número de capas de message passing
- `dropout`: dropout interno del modelo
- `embedding_dim`: tamaño de embeddings categóricos de nodos
- `graph_embedding_dim`: tamaño de embeddings categóricos globales
- `readout_dim`: dimensión del MLP final antes de las cabezas de salida

#### `GNN.training`
Define el loop de entrenamiento, no el modelo ni el optimizador.

- `epochs`: número de épocas
- `batch_size`: número de grafos por batch disjunto
- `weight_decay`: penalización L2 manual sobre pesos densos del modelo
- `shuffle_train`: si `true`, baraja el split train en cada época
- `val_interval_epochs`: cada cuántas épocas se recalcula validación

#### `GNN.optimizer`
Configura el optimizador. El nombre y sus hiperparámetros van juntos para evitar redundancia con `GNN.training`.

##### Campos comunes
- `name`: `adam`, `sgd` o `rmsprop`
- `learning_rate`: tasa de aprendizaje
- `clipnorm`: clipping global por norma; `null` para desactivar
- `clipvalue`: clipping por valor; `null` para desactivar

##### Si `name = adam`
- `beta_1`
- `beta_2`
- `epsilon`
- `amsgrad`

##### Si `name = sgd`
- `momentum`
- `nesterov`

##### Si `name = rmsprop`
- `rho`
- `momentum`
- `epsilon`
- `centered`

#### `GNN.runtime`
Contiene solo la parte operativa de dispositivo y progreso.

- `device`: `gpu` o `cpu`
- `gpu_index`: índice de la GPU si `device = gpu`
- `memory_growth`: activa crecimiento dinámico de memoria en GPU
- `mixed_precision`: intenta usar `mixed_float16` cuando la GPU lo soporta
- `enable_xla`: activa XLA/JIT de TensorFlow
- `batch_progress`: muestra barra de progreso por batch
- `batch_log_interval`: refresca la barra cada N batches

#### Resumen de diseño
La sección `GNN` queda separada así para evitar redundancias:

- **modelo** en `GNN.model`
- **entrenamiento** en `GNN.training`
- **optimizador** en `GNN.optimizer`
- **runtime** en `GNN.runtime`

Así no se repiten campos como `learning_rate`, `epochs` o `device` en varios sitios a la vez.



## Esquema de config recomendado (base 0.6.56 reversionada)

La configuración recomendada queda organizada así:

- `LOT`: identidad lógica del lote
- `STORAGE`: persistencia de formatos de modelo
- `RESOURCE_MANAGER`: política global de RAM
- `GENERATION.runtime`: paralelización de generación (`jobs`, `worker_max_tasks`, hilos TF)
- `BENCHMARK.signature`: firma lógica del benchmark
- `BENCHMARK.runtime`: ejecución del benchmark (`runtime`, `jobs`, afinidad CPU, `worker_max_tasks`)
- `DATASET`: exportación del dataset (sin selección de runtime; se exporta siempre sobre TFLite)
- `SPLIT`: split train/val/test
- `GNN`: entrenamiento de la GNN

La política de RAM se define una sola vez en `RESOURCE_MANAGER`. Las etapas consumen esas políticas globales y solo conservan en sus bloques `runtime` los parámetros propios de scheduling/ejecución.

Defaults recomendados:

- `STORAGE.persist_float_model = false`
- `BENCHMARK.runtime.runtime = "tflite"`
