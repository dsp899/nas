## 0.22.6

- plotting NAS por epoch: `cumulative_max_accuracy` y `cumulative_min_accuracy` pasan a representar la media acumulada de los máximos y mínimos por epoch
- se añaden `cumulative_best_max_accuracy` y `cumulative_best_min_accuracy` para reflejar el mejor techo y el mejor suelo históricos
- plotting NAS por epoch: `rolling_max_accuracy` y `rolling_min_accuracy` pasan a representar la media móvil de los máximos y mínimos por epoch
- se añaden `rolling_best_max_accuracy` y `rolling_best_min_accuracy` para reflejar el mejor techo y el mejor suelo dentro de la ventana móvil


## 0.22.5
- Fixed NAS plotting epoch metrics so cumulative/rolling max, min, mean and median columns are computed with the names expected by the plotting layer.
- `run.py nas plot` no longer fails on missing `cumulative_max_accuracy` when reading an existing search summary.


## 0.22.1
- Rediseñado el plotting NAS para enfatizar gráficas sobre tablas, manteniendo CSVs como trazabilidad interna.
- Añadidas curvas rolling y acumuladas de max, min, mean y median sobre accuracy a nivel de muestra y epoch.
- Añadida gráfica específica de pérdida del controller, separada del resto del análisis.
- Enriquecidos los perfiles por dimensión para incluir min, median, mean y max por valor.

## 0.22.0
- NAS: el sampler ya no trata como agotamiento real el caso en que el controller no descubre novedades dentro del presupuesto de intentos; ahora usa un presupuesto mayor y configurable, con fallback uniforme para evitar paradas prematuras por colapso del controller.
- NAS: nuevos parámetros de controlador `sampling_attempts_multiplier` y `sampling_attempts_minimum` en defaults/config.
- NAS: logging de terminal ampliado con intentos de muestreo, duplicados, uso de fallback y distribución de arquitecturas por número de capas en cada epoch y de forma acumulada.
- NAS plotting rediseñado: análisis estadístico de dimensiones del search space, importancia relativa por `eta_squared`, interacciones por pares, perfiles por valor y plots de progreso/controlador más compactos y significativos.

## 0.21.1
- NAS: stop cleanly when the sampler returns zero new candidates instead of reducing empty arrays in the final epochs.

## 0.21.0

- Unifica los entrypoints públicos en un único `run.py` con sintaxis `python3 run.py <cnn|rnn|nas> <operation>`.
- Elimina `run_cnn.py`, `run_rnn.py`, `run_search.py` y `run_plot_search.py`.
- Corrige el seguimiento global de arquitecturas válidas en NAS: `valid_seen` y `valid_remaining` ya se actualizan entre epochs al reutilizar correctamente `seen_sequences`.
- Reduce el ruido de TensorFlow en terminal configurando el silenciamiento de logs desde el arranque del paquete y del runtime.

## 0.20.0
- NAS runs now write per-run artifacts under `nas/searches/<search_signature>/runs/<run_id>/` instead of appending to a single shared CSV/log directory.
- NAS terminal output now reports global search progress, valid search-space size, and concise per-candidate summaries.
- Inner RNN training progress bars are suppressed during NAS candidate evaluation to reduce terminal noise.

## 0.19.3
- Corregido NAS logger: `memory_mode` se registra desde `config["architecture"]` en vez de `config["data"]`, alineado con el rediseño actual.

## 0.19.2
- Corrige una regresión en `cnn_data.py` al entrenar CNN: el dataset builder volvía a acceder a `self.augmentation.enabled` en vez de `self.preprocess.augmentation_enabled`, provocando `AttributeError` en `run_cnn.py train`.


## 0.19.1
- El default del search space CNN de NAS vuelve a incluir todos los backbones soportados (`vgg16`, `resnet50`, `inceptionV3`) en lugar de fijarse a uno solo.


## 0.19.0
- NAS por defecto alineado con la CNN/RNN base: el search space CNN arranca con un único backbone por defecto (`inceptionV3`) para que el flujo sin JSON use las features realmente exportadas.
- `default_nas_experiment()` ahora sincroniza `data.cnn` con el `search_space.cnn` efectivo.
- El tag del extractor CNN ya no incluye `ft1/ft0`; pasa a usar el formato `<backbone>_fd<feature_dim>`.
- Los runs de NAS cuelgan directamente de la firma del experimento, sin `nas_search_tag` intermedio en el path.
- Mensajes de error de resolución de features actualizados al CLI nuevo basado en `run_cnn.py export_features`.

- 0.18.4: corregida la resolución de features CNN desde RNN/NAS para aceptar el formato real serializado por export_features (`preprocess` y/o `model.extractor`), evitando falsos FileNotFound cuando las features sí estaban registradas.
## 0.18.2
- eliminada la constante muerta `TEST_SPLITS` de `config/supported/shared_supported.py`.
- eliminado su reexport en `config/supported/__init__.py`.

## 0.18.0
- proyecto revisado para tomar Python 3.8 como base objetivo en la rama actual, eliminando sintaxis y patrones incompatibles con 3.10+.
- corrección del arranque por defaults de CNN, RNN y NAS para que la serialización `to_dict()` y los loaders de config tengan la misma estructura efectiva.
- `CnnExperimentConfig.to_dict()` pasa a emitir `training.optimizer` y `training.scheduler` anidados, alineados con los ficheros JSON de ejemplo y con `load_cnn_config()`.
- `RnnExperimentConfig.to_dict()` pasa a emitir `model.decision` anidado, `feature_source.cnn` y `training.allow_epoch_extension_resume`, alineado con `load_rnn_config()` y con el arranque sin JSON.
- `cnn_config_io.py`, `rnn_config_io.py` y `nas_config_io.py` pasan a usar imports perezosos de runtime/train/test/deploy para que la carga de configuración no dependa de TensorFlow antes de tiempo.
- `common/runtime.py` pasa a importar TensorFlow de forma perezosa, reduciendo fallos de importación temprana cuando solo se quiere cargar configuración o defaults.
- añadida normalización interna de payloads de entrenamiento/modelo para evitar incoherencias entre defaults tipados, `to_dict()` y JSONs parciales.
- verificado en local el arranque estático de `run_cnn.py`, `run_rnn.py`, `run_search.py` y la carga sin JSON de `load_cnn_config()`, `load_rnn_config()` y `load_nas_config()`.

## 0.17.0
- documentación unificada en un único README actualizado a la arquitectura de la versión 0.16.x
- eliminación de markdowns redundantes de CNN, RNN, NAS, deploy e integración antigua
- README reescrito para reflejar el arranque unificado, el sistema de configuración, el contrato CNN -> RNN/NAS y la estructura real del proyecto

## 0.16.0
- unificación del arranque de CNN, RNN y NAS con `--config` opcional
- si no se pasa fichero JSON, cada subsistema arranca desde sus defaults internos
- eliminación de payloads por defecto manuales duplicados en los loaders de config
- introducción de `default_rnn_experiment()` y `default_nas_experiment()` para alinear la construcción de defaults tipados

# Changelog

## 0.15.0
- refactor agresivo completo de la estructura interna para usar módulos reales `cnn_*`, `rnn_*` y `common/*` sin wrappers heredados
- fusión de `rnn_export` y `rnn_tflite_eval` en `rnn_deploy`
- migración de `cnn_dataset` a `cnn_data`, `cnn_backbones` a `cnn_model`, `cnn_training` a `cnn_train`
- migración de `rnn_data_pipeline` a `rnn_data` y extracción del núcleo de modelo a `rnn_model`
- eliminación de `app_*`, `deploy_tracking`, `config_files`, `model_io_utils` y `optimizer_utils`
- eliminación de compatibilidades internas con la nomenclatura antigua
## 0.17.1
- Compatibilidad con Python 3.8 en type hints de la capa de configuración y deploy (`Optional`, `Union`, `Tuple`, `Set` en lugar de sintaxis de Python 3.10+).
- Sin cambios funcionales de pipeline; parche de arranque/importación para entornos como el tuyo.
## 0.18.3
- Fixed missing typing imports in `rnn_model.py` and `rnn_deploy.py` for Python 3.8 imports.
- Fixed `rnn_train.py` to import `_next_state_for_next_clip` from `rnn_model.py`.


## 0.22.2
- Corregido `ImportError` en NAS al alinear `nas_controller` con la API actual de `common.training` (`build_keras_optimizer` en lugar de `make_optimizer`).
- Corregido el estado de `ReduceLROnPlateau` del controller NAS para usar `ReduceLrPlateauState`.
## 0.22.3
- Corregido el controller NAS para evitar un error de máscara en Keras al construir la salida softmax en TensorFlow/Keras de Python 3.8.


- 0.22.4: plotting NAS separado para que cada figura contenga un único plot; accuracy y epoch divididos en acumulado/rolling, distribución de capas dividida en normal/acumulada, perfiles por dimensión separados en frecuencia y accuracy.

## 0.22.7
- Rediseño de `dimension_profiles` en el análisis NAS: ahora cada dimensión genera un único gráfico unificado por epoch con dos ejes Y, mostrando la accuracy media por valor (eje izquierdo) y la acumulación de selecciones por valor (eje derecho).
- Se sustituyen las figuras separadas `*_frequency.png` y `*_accuracy.png` por `*_epoch_profile.png` para cada dimensión.

## 0.22.8
- Rediseño de `dimension_profiles` en el plotting NAS: por defecto cada dimensión genera cuatro gráficas con las cuatro combinaciones 2x2 entre accuracy acumulada (`cumulative_mean_accuracy`, `cumulative_median_accuracy`) y selección acumulada (`cumulative_selection_count`, `cumulative_selection_share`).
- Los perfiles por dimensión ahora reflejan una relación temporal más coherente entre evolución acumulada de accuracy y evolución acumulada de selección por valor en función del `search_epoch`.

## 0.22.9
- Rediseño de la estructura de `analysis/` en el plotting NAS con directorios claros: `tables/`, `overview/`, `search_space/` y `dimensions/`.
- Eliminadas las gráficas globales `search_accuracy_cumulative.png` y `search_accuracy_rolling.png` del análisis NAS.
- Los perfiles por dimensión ahora se generan en subdirectorios propios dentro de `analysis/dimensions/<dimension>/`.
- Añadidos por dimensión: `value_trajectory_summary.csv`, `selection_accuracy_alignment.png` y `value_dynamics.png`, además de las cuatro combinaciones acumuladas entre accuracy (`mean`, `median`) y selección (`count`, `share`).

## 0.22.10
- Corregido `run.py nas plot` para que imprima la estructura nueva del análisis NAS y deje de referenciar atributos eliminados (`accuracy_cumulative_plot` y `accuracy_rolling_plot`).

## 0.22.11
- Rediseño del análisis NAS con una nueva sección `analysis/correlations/` dedicada a correlaciones selección-rendimiento.
- Eliminados los plots ambiguos `selection_accuracy_alignment.png` y `value_dynamics.png` por dimensión.
- Añadidos: correlación evolutiva por valor dentro de cada dimensión, matrices de correlación entre valores para accuracy y selección, y un resumen global de `selection vs final performance` por dimensión.
- Se mantiene `analysis/dimensions/<dimension>/` para los cuatro perfiles acumulados y `value_trajectory_summary.csv`.

## 0.22.12
- Corrección de `nas_plotting.py`: se restaura el helper interno `_plot_dimension_profile` usado por los perfiles por dimensión y la nueva sección de correlaciones, evitando el `NameError` al ejecutar `run.py nas plot`.

## 0.22.13
- Reestructuración de `analysis/correlations/` en el plotting NAS: se elimina `within_dimension/` y se reemplaza por dos ramas claras, `by_value_cumulative/` y `by_value_local/`.
- Las correlaciones by value se calculan ahora en dos variantes: sobre métricas acumuladas y sobre métricas locales por epoch.
- Se añade un umbral mínimo de `MIN_VALID_EPOCHS_FOR_CORRELATION=3` para que las correlaciones solo se calculen cuando un valor aparece en suficientes epochs válidos.
- Se mantienen los profiles acumulados por dimensión y se añaden resúmenes `value_trajectory_summary_local.csv` junto a los acumulados.

## 0.22.14
- Reestructuración de `analysis/correlations/` en el análisis NAS: se eliminan las correlaciones `within_dimension` y se organizan en `by_value_cumulative/`, `by_value_local/`, `by_dimension_over_time/` y `final_dimension_alignment/`.
- Añadidas correlaciones por dimensión y por epoch, tanto `cumulative` como `local`, con gráficas separadas para mean y median.
- Las correlaciones by-value acumuladas y locales se calculan solo para valores que aparecen en al menos `MIN_VALID_EPOCHS_FOR_CORRELATION` epochs válidos.

## 0.22.15
- Añadido `README_ANALYSIS.md` con explicación detallada de toda la carpeta `analysis/` del plotting NAS: estructura, artefactos fuente, significado de cada reporte y guía de interpretación.
- `README.md` actualizado para enlazar a `README_ANALYSIS.md` desde la sección de análisis NAS.
