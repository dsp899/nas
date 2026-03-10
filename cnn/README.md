# Clasificación de acciones en vídeo con CNN sobre UCF101

## Entorno de trabajo

Este proyecto está implementado en **Python** y utiliza **TensorFlow/Keras** para entrenar, evaluar y reutilizar redes convolucionales preentrenadas sobre el dataset **UCF101**.

La versión actual del código trabaja con:
- `tf.keras`
- políticas de precisión mixta (`mixed_float16`)
- lectura de vídeo con OpenCV
- almacenamiento de modelos en formato Keras/HDF5 (`.h5`)
- almacenamiento de salidas y características en formato NumPy (`.npy`)

### Dependencias principales

- Python 3
- TensorFlow
- NumPy
- OpenCV (`cv2`)
- Pandas

---

## Gestión del dataset

Puedes acceder al dataset [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) desde el enlace oficial.

La estructura de directorios esperada por el proyecto es la siguiente:

```bash
./data/ucf101/videos
./data/ucf101/names
````

### Organización esperada

* `./data/ucf101/videos`: contiene los vídeos del dataset, organizados en carpetas, una por cada clase de acción.
* `./data/ucf101/names`: contiene los archivos de partición de entrenamiento y test del dataset.

La clase `UCF101` implementa la lógica de:

* selección de subconjuntos de acciones,
* lectura de vídeos disponibles,
* mapeo de clases globales y locales,
* construcción de particiones de entrenamiento y test,
* preparación de estructuras listas para el procesamiento posterior.

### Subconjuntos soportados

Actualmente el código contempla distintos subconjuntos de clases, entre ellos:

* `pmi`, `pmi50`
* `bm`, `bm50`
* `hoi`, `hoi50`
* `hhi`, `hhi50`
* `sports`, `sports50`
* `all`, `all50`

Por defecto, la ejecución trabaja con el **split 1** del dataset (`split01`).

---

## Carga y procesado de vídeo

La lógica de lectura y preparación de datos se concentra en:

* `ucf101.py`: definición del dataset y extracción de frames
* `run_cnn.py`: script principal de ejecución

La clase `Frames` se encarga de:

* leer vídeos con OpenCV,
* extraer un número fijo de frames equiespaciados,
* redimensionarlos con `tf.image.resize_with_pad`,
* convertirlos a RGB,
* construir datasets de TensorFlow listos para entrenamiento, evaluación o extracción de características.

### Estrategia de extracción de frames

Para cada vídeo:

1. se obtiene el número total de frames,
2. se selecciona un conjunto de frames equiespaciados,
3. cada frame se redimensiona al tamaño de entrada de la red,
4. si algún frame no puede leerse, se sustituye por un frame negro.

Esto permite tratar vídeos de distinta duración con una entrada homogénea.

---

## Gestión de la CNN

El proyecto realiza **fine-tuning** de una CNN preentrenada sobre ImageNet para clasificación de acciones a partir de frames.

Las arquitecturas contempladas en el proyecto son:

* `vgg16`
* `resnet50`
* `inceptionV3`

La construcción del modelo se realiza en `cnn.py`.

Sobre la base convolucional preentrenada se añaden:

* una capa `GlobalAveragePooling2D` llamada `feature_extractor`,
* una capa `Dropout`,
* una capa densa de 2048 unidades con activación ReLU,
* otra capa `Dropout`,
* una capa final `Dense` con activación `softmax`.

### Hiperparámetros por defecto

La configuración por defecto definida en `config.py` incluye:

* `batch = 16`
* `epochs = 10`
* `learning_rate = 0.001`
* `dense_units = 2048`
* `dropout = 0.5`

Además, el proyecto distingue entre:

* número de frames usados para entrenamiento/evaluación del clasificador,
* número de frames usados para extracción de características.

### Uso de GPU y CPU

El proyecto incluye configuración automática de dispositivo:

* si hay GPU disponible, se selecciona la GPU indicada y se activa `memory_growth`,
* si no hay GPU, TensorFlow se configura para CPU usando los núcleos disponibles,
* se activa la política global `mixed_float16`.

---

## Operaciones disponibles

El script principal del proyecto es:

```bash
python3 run_cnn.py
```

Este script admite tres operaciones claramente diferenciadas:

* `train`: entrena una CNN preentrenada sobre UCF101
* `eval`: evalúa un modelo ya entrenado sobre test
* `predict`: extrae características usando un modelo ya entrenado

La forma general de uso es:

```bash
python3 run_cnn.py --operation <train|eval|predict> --cnn <arquitectura> --data <dataset> --frames <n> --size <s> --gpu <id>
```

Aunque las tres operaciones comparten los mismos argumentos, **no todos los parámetros significan exactamente lo mismo en los tres casos**, especialmente `--frames`.

---

## Parámetros de ejecución

### `--operation`

Indica la operación a ejecutar.

Valores soportados:

* `train`
* `eval`
* `predict`

---

### `--cnn`

Arquitectura CNN utilizada.

Valores documentados en el proyecto:

* `vgg16`
* `resnet50`
* `inception_v3`

### Observación importante

En el código hay una pequeña inconsistencia de nombres:

* en unos puntos aparece `inception_v3`,
* en otros aparece `inceptionV3`.

Conviene mantener esta nota en mente al ejecutar experimentos.

---

### `--data`

Subconjunto del dataset UCF101 sobre el que se trabaja.

Ejemplos:

* `all`
* `all50`
* `pmi`
* `pmi50`
* `bm`
* `bm50`
* `hoi`
* `hoi50`
* `hhi`
* `hhi50`
* `sports`
* `sports50`

Este parámetro debe coincidir con el subconjunto usado al entrenar el modelo que posteriormente se quiera evaluar o reutilizar.

---

### `--size`

Tamaño espacial de entrada de la CNN.

Ejemplo:

```bash
--size 299
```

Este valor forma parte del experimento y debe coincidir entre entrenamiento, evaluación y extracción de características.

---

### `--frames`

Este parámetro requiere una aclaración importante porque **su interpretación depende de la operación**.

#### En `train`

`--frames` indica el **número de frames por vídeo usados para entrenar el modelo**.

Ejemplo:

```bash
--frames 15
```

En ese caso, el entrenamiento se realiza usando 15 frames equiespaciados de cada vídeo.

#### En `eval`

En `eval`, `--frames` no debe interpretarse como:

> número de frames con el que quiero evaluar ahora

sino como:

> número de frames con el que fue entrenado el modelo que quiero cargar

Es decir, este parámetro sirve principalmente para identificar el experimento y localizar el modelo correcto.

El número efectivo de frames usados internamente en evaluación depende de la configuración definida en `config.py`.

#### En `predict`

En `predict` ocurre lo mismo que en `eval`.

Aquí `--frames` indica:

> número de frames con el que fue entrenado el modelo del que quiero extraer características

No representa necesariamente el número real de frames sobre los que se extraerán las *features*, ya que ese valor depende de la configuración interna del proyecto.

---

### `--gpu`

Identificador de GPU a utilizar.

Ejemplo:

```bash
--gpu 0
```

Si no hay GPU disponible, el proyecto pasa a ejecución en CPU.

---

## Operación 1: entrenamiento (`train`)

La operación `train` entrena una CNN preentrenada sobre el subconjunto seleccionado de UCF101.

### Ejemplo de uso

```bash
python3 run_cnn.py --operation train --cnn vgg16 --data pmi50 --frames 15 --size 299 --gpu 0
```

### Qué hace esta operación

* carga el conjunto de entrenamiento y test,
* extrae de cada vídeo el número de frames indicado por `--frames`,
* construye la arquitectura CNN seleccionada,
* realiza el proceso de *fine-tuning*,
* valida el modelo sobre el conjunto de test,
* guarda checkpoints del entrenamiento.

### Cómo interpreta `--frames`

En `train`, `--frames` significa literalmente:

> número de frames por vídeo usados para entrenar la CNN

### Estrategia de entrenamiento

Durante el entrenamiento:

* se usa `EarlyStopping` monitorizando `val_loss`,
* se guarda el mejor modelo,
* se guarda también el último estado entrenado,
* se registra información del experimento en un archivo JSON.

### Ficheros generados

Los resultados del entrenamiento se almacenan en una ruta del tipo:

```bash
./models/keras/{dataset}/cnn/{arquitectura}/
```

Dentro de esa carpeta se generan típicamente:

* `*_best.h5`: mejor modelo según rendimiento en validación
* `*_last.h5`: último estado del entrenamiento
* `*_info.json`: metadatos y progreso del entrenamiento

---

## Operación 2: evaluación (`eval`)

La operación `eval` se utiliza para evaluar en test un modelo ya entrenado.

### Ejemplo de uso

```bash
python3 run_cnn.py --operation eval --cnn vgg16 --data pmi50 --frames 15 --size 299 --gpu 0
```

### Qué hace esta operación

* reconstruye el conjunto de test,
* carga un modelo previamente entrenado,
* ejecuta la evaluación con `model.evaluate(...)`,
* muestra por pantalla la pérdida y la precisión.

### Qué modelo carga

En la implementación actual, `eval` carga el modelo:

```bash
*_best.h5
```

Es decir, **se evalúa el mejor checkpoint guardado durante el entrenamiento**, no el último.

### Cómo interpreta `--frames`

En `eval`, `--frames` debe coincidir con el valor usado cuando se entrenó el modelo.

Debe entenderse como:

> quiero evaluar el modelo que fue entrenado con este número de frames

No significa necesariamente:

> quiero usar ahora exactamente este número de frames para evaluar

### Número real de frames usados en evaluación

El número efectivo de frames usados durante la evaluación depende de la configuración interna definida en `config.py`.

Por tanto:

* `--frames` identifica el modelo que se carga,
* la configuración interna determina cuántos frames se usan al construir los datos de evaluación.

---

## Operación 3: extracción de características (`predict`)

La operación `predict` permite reutilizar un modelo ya entrenado para extraer representaciones intermedias de los vídeos.

### Ejemplo de uso

```bash
python3 run_cnn.py --operation predict --cnn vgg16 --data pmi50 --frames 15 --size 299 --gpu 0
```

### Qué hace esta operación

* carga el conjunto de entrenamiento y test,
* carga un modelo ya entrenado,
* reconstruye un extractor usando la capa `feature_extractor`,
* genera vectores de características,
* guarda en disco las *features*, las etiquetas y los identificadores de vídeo.

### Qué modelo carga

En la implementación actual, `predict` también carga:

```bash
*_best.h5
```

Después de cargar ese modelo, se construye un nuevo modelo cuya salida es la capa:

```bash
feature_extractor
```

### Cómo interpreta `--frames`

En `predict`, `--frames` debe entenderse como:

> número de frames con el que fue entrenado el modelo que quiero reutilizar

Es decir, sirve para identificar y cargar el modelo correcto.

### Número real de frames usados para extraer características

El número efectivo de frames sobre los que se generan las *features* está gobernado por la configuración definida en `config.py`, no necesariamente por el valor pasado en `--frames`.

---

## Salida de la extracción de características

Los ficheros generados por `predict` se almacenan en una ruta del tipo:

```bash
./data/features/{dataset}/{cnn}/{nombre_experimento}/
```

Se generan archivos como:

* `features_train_*.npy`
* `labels_train_*.npy`
* `video_id_train_*.npy`
* `features_test_*.npy`
* `labels_test_*.npy`
* `video_id_test_*.npy`

Estas salidas pueden reutilizarse en experimentos posteriores.

---

## Resumen práctico de las tres operaciones

### Usa `train` cuando quieras

* entrenar un nuevo modelo,
* crear checkpoints del experimento,
* generar un clasificador nuevo para una combinación concreta de arquitectura, dataset, tamaño y número de frames de entrenamiento.

### Usa `eval` cuando quieras

* medir el rendimiento de un modelo ya entrenado,
* evaluar sobre test el checkpoint `best`.

### Usa `predict` cuando quieras

* reutilizar el mejor modelo entrenado,
* extraer *features* de la capa `feature_extractor`,
* guardar representaciones en formato NumPy para otras etapas del pipeline.

---

## Flujo de trabajo recomendado

1. Descargar y organizar el dataset UCF101 en las carpetas esperadas.
2. Ejecutar `train` para entrenar la CNN deseada.
3. Ejecutar `eval` para medir el rendimiento sobre test.
4. Ejecutar `predict` para extraer características del modelo entrenado.

---

## Estructura general del proyecto

```bash
.
├── config.py
├── cnn.py
├── run_cnn.py
├── ucf101.py
├── README.md
├── data
│   ├── ucf101
│   │   ├── videos
│   │   └── names
│   └── features
└── models
    └── keras
```

---

## Notas importantes

* El proyecto actual trabaja fundamentalmente a nivel de **frames**.
* La operación `predict` no realiza clasificación final, sino extracción de características intermedias.
* Tanto `eval` como `predict` cargan el checkpoint `*_best.h5`.
* El valor `--frames` en `eval` y `predict` debe interpretarse como el número de frames con el que se entrenó el modelo que se quiere cargar.
* El número efectivo de frames usados en evaluación y extracción depende de la configuración interna del proyecto.
* El split utilizado actualmente es `split01`.
* Existen pequeñas inconsistencias de nombres en el código, propias de una refactorización parcial, por lo que conviene mantener coherencia al nombrar experimentos y arquitecturas.

---

## Ejemplos completos de uso

### Entrenamiento

```bash
python3 run_cnn.py --operation train --cnn resnet50 --data all50 --frames 15 --size 299 --gpu 0
```

### Evaluación

```bash
python3 run_cnn.py --operation eval --cnn resnet50 --data all50 --frames 15 --size 299 --gpu 0
```

### Extracción de características

```bash
python3 run_cnn.py --operation predict --cnn resnet50 --data all50 --frames 15 --size 299 --gpu 0
```

