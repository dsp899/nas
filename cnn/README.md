## Entorno de trabajo
Python 3.6.9

Pip 21.3.1

Tensorflow 1.15.0

## Gestión del Dataset

Puedes acceder al dataset [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) desde el enlace proporcionado.

La estrcutura de directorios debe ser la siguiente:
- `./data/ucf101/videos`: aquí se han de almacenar todos los videos tras descargarlos. Todavía deben estar ordenads por carpetas, cada carpeta representando una acción. [Videos UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
- `./data/ucf101/names`: aquí deben almacenarse los ficheros que especifican que videos pertenecen al conjunto de videos para entrenamiento y qué videos pertenecen al conjunto de videos para test. [Train/Test files](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip) 

![Estructura de directorio para los datos](pictures/data_directories.png)
Los scripts python con los que se gestiona la carga de video a memoria, su procesado y su posterior almacenamiento en disco como archivos *numpy* son *manage_data.py* y *load_data.py*.

Para cargar los datos de vídeos, procesarlos y guardarlos como frames en formato *numpy*, puedes utilizar el siguiente comando en Python:

```bash
python3 manage_data.py --name pmi50 --mode train --frames 15 --rescaled_size 299 --size 299
```
### Parámetros del Comando:
- `--name`: Este parámetro permite especificar las categorías que se desean utilizar del dataset. Aunque el dataset contiene 101 categorías, estas se encuentran agrupadas en 5 grandes grupos (*Human-Object Interaction, Body-Motion Only, Human-Human Interaction, Playing Musical Instruments, Sports*). Además, el dataset UCF101 es una extensión del dataset UCF50 que contenía 50 categorías agrupadas del mismo modo. Se puede utilizar este parámetro para seleccionar unicamente alguno de estos grupos de acciones. Por ejemplo, se puede utilizar `--name pmi` para trabajar con el grupo de acciones *Playing Musical Instruments*. En la fase de pruebas, para reducir el tamaño del dataset, se puede trabajar con los grupos del UCF50 utilizando, por ejemplo, `--name pmi50`.

- `--mode`: Este parámetro permite escoger entre procesar el conjunto de videos del entrenamiento `--mode train` o el conjunto de videos de test `--mode test`. 
El dataset contiene 13320 videos en total. Para evitar aleatoreidad en los futuros experimentos que se llevasen a cabo, sus creadores especificaron tres posibles divisiones en *train/test*. En el *script load_data_back_up.py* esta fijado el *split 1*. 

- `--frames`: Este parámetro permite escoger el número de frames total que se extraeran y procesaran en cada video. Se puede usar `--frames 15`,`--frames 20`, etc. Hay que tener cuidado con la cantidad de memoria RAM disponible. 

    Cada video puede durar entre 2 y 36 segundos aproximadamente, durando de media entorno a los 15 segundos. Todos fueron grabados a 25 FPS de forma que la cantidad de frames por video disponble es bastante grande. Entonces este parametro es el numero de frames que quiero extraer de cada video. He trabajado con 15 frames por video por temas de memoria RAM pero si se programa de manera más eficiente se puede trabajar con mas frames por video. Los frames extraidos de un video son equidistantes entre si, de forma que abarcan la totalidad del video, dure 2 o 36 segundos. 

- `--rescaled_frames`:  Este argumento permite hacer un resize de todos los frames para conseguir frames de tamaño cuadrado. Se puede usar por ejemplo `--rescaled_size 299`. Todos los frames de cada video tienen una resolución nativa de 320x240 pixeles.  

    Para hacer pruebas es mejor hacer un resize de menos resolucion para ahorrar espacio en la memoria RAM y ganar en velocidad a la hora de entrenar las redes.

- `--size`: Este argumento sirve para recortar cada frame respecto a su centro. Se puede usar por ejemplo `--size 224`. Esto significa que cada frame terminara con una resolucion de 224x224 respecto a su centro.

    Es determinadas ocasiones es beneficioso hacer un *"crop center"*de los frames. Sin embargo, he estado trabajndo sin *"crop center"* con el parámetro puesto a `--size 299`
    El valor que se pase al parámetro ha de ser siempre menor al valor de resize del argumento `--rescaled_size`

## Gestión de la CNN
El dataset UCF101 no es lo suficientemente grande para entrenar la red desde cero. Apenas se compone de 101 clases y no tiene demasiados videos por clase. Esto provoca que haya overfitting o mala generalizacion de la red a la hora de entrenarla desde cero. Para solventar este problema es importante utilizar una red preentrenada en un conjunto de imagenes o videos mucho más grande. 

En este caso, he utilizado redes preentrenadas en el dataset Imagenet. Este dataset contiene 1.200.000 imagenes, pertenecientes a 1000 clases. Estas redes prentrenadas están disponibles como clases del *framework* Tensorflow y se encuentran en el módulo *tf.keras.applications*.

Los scripts python con los que se gestiona la creación, el entrenamiento *(fine tunning)*, evaluación e inferencia *(feature extraction)* de las CNN son *manage_cnn.py* y *load_model_cnn.py*.

## Fine tunning
Para cargar la CNN preentrenada en Imagenet, hacerle *fine tunning* y guardar el modelo en formato *SavedModel* de tensorflow, puedes utilizar el siguiente comando en Python:
```bash
python3 run_cnn.py --operation train --cnn vgg16 --data pmi50 --frames 15 --size 299
```
Por defecto, los modelos de tensorflow se guardaran en la carpeta *./built/tensorflow_savedModels*

Para cargar la CNN entrenada (*fine tunning*) con el UCF101, y evaluarala sobre el conjunto de test obteniendo por pantalla el accuracy de la red, puedes utilizar el siguiente comando en Python:
```bash
python3 run_cnn.py --operation eval --cnn vgg16 --data pmi50 --frames 15 --size 299
```

### Parámetros de ambos Comandos:
- `--operation`: Este argumento permite elgir que accion realizar con la red. Para entrenar (*fine-tunning*) se utiliza el parametro `--operation train` y para evaluar la red se usa `--operation eval`.
- `--model`: Este argumento especifica la aruitectura de red, es decir, la clase Python elegida para hacer el *fine tunning*. Se puede usar con `--model vgg16` o `--model resnet50` a falta de añadir más arquitecturas de CNNs.
- `--data`, `--frames` y `--size`: Sirven para detallar el dataset de frmaes alamcenados en disco con el que se va llevar a cabo la operación elegida. Támbien se usar en la estructura de directorios para almacenar las redes en el formato *SavedModel* de Tensorlfow.

## Feature Extraction
Para generar el *dataset* de entrenamiento para la red LSTM se utiliza la técnica de *transfer learning* conocida como *feature extraction*.  Esta técnica consiste en generar las *features*  correspondientes a los frames tanto del conjunto de datos de entrenamiento como de test. 

Las *features* de un frame es una imagen con una dimension reducida que representan alguna información importante del frame original. Cuando se realiza la inferencia sobre un frame o imagen, cada capa convolucional de la red CNN presenta a su salida una *feature* centrada en algun tipo de información sobre el frame original, siendo esta información más detallada en la salida de las capas más profundas de la red. 

Las *features* que genero para entrenar la red LSTM las obtengo de la última capa de convolución de la red CNN escogida. Para generar estas features y almacenarlas en disco, puedes utilizar el siguiente comando en Python:

```bash
python3 run_cnn.py --operation predict --cnn vgg16 --data pmi50 --frames 15 --size 299
```

### Parámetros de ambos Comandos:
- `--operation`: Este argumento permite elgir que accion realizar. Para realizar un *feature extraction* del *dataset* se utiliza el parametro `--operation infer`.
- `--infer`: Este argumento se usa para escoger el conjunto de datos sobre el que realizar el *feature extraction*. Se puede usar `--infer train` o `--infer test`.
- `--model`: Este argumento especifica la red CNN con la que se extraeran las *features* del *dataset*. Se puede usar con `--model vgg16` o `--model resnet50` a falta de añadir más arquitecturas de CNNs.
- `--data`, `--frames` y `--size`: Sirven para detallar el dataset de frames alamcenados en disco con el que se va llevar a cabo la operación elegida.