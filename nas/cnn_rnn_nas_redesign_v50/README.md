# CNN-RNN NAS

Versión **0.17.0**.

Esta es la documentación unificada del proyecto. A partir de esta versión, la referencia principal es **este README** y el histórico de cambios vive en `CHANGELOG_SYNC_RNNBENCHLIB.md`.

## Qué hace el proyecto

El flujo completo queda dividido en tres bloques:

- **CNN**: entrena un extractor por frame y un clasificador, y puede exportar **features persistidas a nivel de vídeo** para alimentar la RNN y la NAS.
- **RNN**: entrena y prueba un modelo temporal sobre las features CNN ya exportadas, y gestiona su deploy TFLite bajo una interfaz única de `deploy`.
- **NAS**: explora el espacio de diseño temporal de la RNN usando las features CNN persistidas como fuente de datos.

## Estructura actual del código

```text
cnn_rnn_nas/
  cnn/
    cnn_config_io.py
    cnn_data.py
    cnn_model.py
    cnn_train.py
    cnn_test.py
    cnn_features.py
    cnn_deploy.py

  rnn/
    rnn_config_io.py
    rnn_data.py
    rnn_model.py
    rnn_train.py
    rnn_test.py
    rnn_deploy.py

  nas/
    nas_config_io.py
    nas_controller.py
    nas_engine.py
    nas_plotting.py
    nas_search_space.py

  common/
    artifacts.py
    config_io.py
    model_io.py
    registries.py
    runtime.py
    training.py

  config/
    cnn_config.py
    rnn_config.py
    nas_config.py
    deploy_config.py
    specs/
    defaults/
    supported/
```

La intención es que **CNN** y **RNN** tengan una separación funcional paralela:

- `*_data`: dato, loaders y consumo
- `*_model`: construcción de modelo
- `*_train`: entrenamiento
- `*_test`: test normal del modelo float
- `*_deploy`: export y test del artefacto de deploy

La excepción natural es `cnn_features.py`, porque la exportación de features solo existe en CNN.

## Lanzadores

### CNN

```bash
python3 run.py cnn train [--config configs/examples/cnn_train.json]
python3 run.py cnn test [--config configs/examples/cnn_train.json]
python3 run.py cnn export_features [--config configs/examples/cnn_train.json]
python3 run.py cnn deploy [--config configs/examples/cnn_train.json]
```

Operaciones públicas CNN:

- `train`
- `test`
- `export_features`
- `deploy`

### RNN

```bash
python3 run.py rnn train [--config configs/examples/rnn_train.json]
python3 run.py rnn test [--config configs/examples/rnn_train.json]
python3 run.py rnn deploy [--config configs/examples/rnn_train.json]
```

Operaciones públicas RNN:

- `train`
- `test`
- `deploy`

### NAS

```bash
python3 run.py nas search [--config configs/examples/nas_search.json]
```

Operación pública NAS:

- `search`

### Análisis de búsquedas NAS

```bash
python3 run.py nas plot --summary-json <ruta_al_summary.json>
```

El análisis NAS genera tablas y gráficas de:
- consulta `README_ANALYSIS.md` para una explicación detallada de la carpeta `analysis/`, sus subdirectorios, las fuentes de datos y la interpretación de cada reporte
- progreso global de la búsqueda
- diagnóstico del controller
- importancia relativa de las dimensiones del search space
- interacciones por pares entre dimensiones
- perfiles por valor para cada dimensión sampleada

## Sistema de configuración

El arranque está unificado en CNN, RNN y NAS.

### Regla general

- `--config` es **opcional**.
- si no pasas JSON, el subsistema arranca desde sus **defaults internos tipados**
- si pasas JSON, el fichero se interpreta como **override parcial** sobre esos defaults

El flujo real es:

```text
JSON opcional -> merge con defaults -> construcción de config tipada -> ejecución
```

### Capas de configuración

- `config/specs/`: estructuras de datos compartidas
- `config/defaults/`: valores por defecto por dominio y subsistema
- `config/supported/`: catálogos de valores soportados
- `config/cnn_config.py`, `config/rnn_config.py`, `config/nas_config.py`: configs tipadas de cada subsistema
- `cnn/cnn_config_io.py`, `rnn/rnn_config_io.py`, `nas/nas_config_io.py`: carga desde fichero y arranque

## Flujo recomendado de punta a punta

```bash
# 1) Entrenar CNN
python3 run.py cnn train --config configs/examples/cnn_train.json

# 2) Exportar features CNN persistidas para RNN/NAS
python3 run.py cnn export_features --config configs/examples/cnn_train.json

# 3) Entrenar o testear la RNN
python3 run.py rnn train --config configs/examples/rnn_train.json
python3 run.py rnn test --config configs/examples/rnn_train.json

# 4) Lanzar una búsqueda NAS
python3 run.py nas search --config configs/examples/nas_search.json

# 5) Export o test de deploy
python3 run.py cnn deploy --config configs/examples/cnn_train.json
python3 run.py rnn deploy --config configs/examples/rnn_train.json
```

## Contrato CNN -> RNN / NAS

La CNN persiste **features a nivel de vídeo**. La RNN y la NAS consumen esas features como fuente canónica.

### Identidad de la fuente CNN

La fuente de features se identifica por:

- `cnn_training_signature`
- `cnn_feature_export_signature`

### Layout de features CNN

```text
artifacts/partitions/
  dataset_<name>_<split>_<partition_mode>/
    cnn/features/
      <cnn_training_signature>/
        <predict_preprocess>/
          <cnn_feature_export_signature>/
```

La RNN ya **no depende de secuencias persistidas** como artifact principal. El consumo temporal se deriva desde las features exportadas por CNN.

## CNN

### Diseño

El bloque CNN se organiza sobre estos ejes:

- `dataset`
- `preprocess`
- `extractor`
- `head`
- `training`
- `runtime`

### Preprocesado

El preprocesado es siempre **personalizado**.

La estrategia temporal está unificada en una sola variable:

- `sampling`

Y lo que puede variar entre entrenamiento y export/predicción es el número de frames:

- `train_frames`
- `predict_frames`

### Extractor

El extractor CNN siempre termina en:

- backbone sin top
- `GlobalAveragePooling2D`
- proyección a `feature_dim`
- salida estable `frame_features`

### Head

El head de clasificación se define por:

- `hidden_units`
- `dropouts`

## RNN

### Espacio de diseño temporal

- `direction`: `unidirectional | bidirectional`
- `memory_mode`: `none | carry_forward`
- `video_decision`
- `video_decision_input`
- tamaño del encoder y del head

### Semántica temporal

- `direction` controla el contexto **intra-clip**
- `memory_mode` controla la memoria **inter-clip**
- en `bidirectional + carry_forward`, solo se propaga la rama **forward** entre clips
- `video_decision` y `video_decision_input` controlan la decisión final a nivel de vídeo

### Deploy RNN

El deploy del RNN agrupa bajo una sola operación tanto la exportación como las pruebas del artefacto TFLite.

`deploy.action` puede tomar estos valores:

- `export`
- `test_runtime`
- `test_pipeline`

## CNN deploy

El deploy del CNN agrupa la exportación y las pruebas relacionadas con el artefacto Vitis AI.

`deploy.action` puede tomar estos valores:

- `export`
- `test_classifier`
- `test_extractor`
- `test_pipeline`

## NAS

### Organización

El bloque NAS queda separado en:

- `data_source`
- `search_space`
- `controller`
- `runtime`

### Search space

La NAS parte del **search space completo soportado por el proyecto**. Ya no se usan presets de search space.

### Controller

La configuración del controller está desacoplada en:

- `model`
- `optimizer`
- `scheduler`
- `training`

## Layout general de artifacts

### CNN

```text
artifacts/partitions/dataset_<name>_<split>_<partition_mode>/cnn/
  models/<train_preprocess>/<backbone>/<head>/<training_signature>/
  features/<cnn_training_signature>/<predict_preprocess>/<cnn_feature_export_signature>/
  deploy/<train_preprocess>/<backbone>/<head>/<training_signature>/<deploy_signature>/
```

### RNN

El RNN consume las features CNN y genera sus propios registros, checkpoints y artifacts de deploy bajo `artifacts/partitions/.../rnn/...`.

### Registries

Los registros SQLite viven bajo:

```text
artifacts/registries/
```

La capa de registro se centraliza en `cnn_rnn_nas/common/registries.py`.

## Configs de ejemplo

Se incluyen ejemplos mínimos en:

- `configs/examples/cnn_train.json`
- `configs/examples/rnn_train.json`
- `configs/examples/nas_search.json`

Puedes usarlos como base y sobrescribir solo lo que cambie.

## Estado de la documentación

A partir de esta versión:

- `README.md` es la documentación funcional principal
- `CHANGELOG_SYNC_RNNBENCHLIB.md` mantiene el histórico
- se han eliminado los markdowns paralelos de CNN, RNN, NAS y deploy para evitar duplicidades
