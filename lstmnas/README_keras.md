## Entorno de trabajo
Python 3.6.9

Pip 21.3.1

Tensorflow 1.15.0


## Gestión de la LSTM

Los scripts python con los que se gestiona la creación, entrenamiento y evaluación de la red LSTM son *manage_lstm.py* y *load_model_rnn.py*

Una vez las *features* del *dataset* se encuentran almacenadas se puede utilizar el siguiente comando  en Python para entrenar la red LSTM:

```bash
python3 run_rnn.py --operation train --rnn lstm --direction bidirectional --units 128,64,32 --cnn vgg16 --data pmi50 --frames 15 --size 299 --seq 3 --state stateless
```

Una vez las la red ha sido entrenada, se puede utilizar el siguiente comando en Python para evaluar la precisión de la red:
```bash
python3 manage_lstm_keras.py --operation eval --cnn inceptionV3 --arch single --direction unidirectional --seq 3 --stride 1 --units 1024 --device gpu --data pmi50 --frames 15 --size 299
```

### Parámetros de ambos Comandos:
- `--operation`: Este argumento permite elegir que accion realizar. Para realizar un entrenamiento se utiliza el parametro `--operation train` y para realizar una evaluación se utiliza `--operation eval`.
- `--model`: Este argumento especifica la arquitectura en red LSTM en cuanto al número de capas. Se puede usar con `--model single` para usar una red de una capa o `--model stacked` para usar una red de dos capas apiladas.
- `--direction`: Este argumento permite elegir el sentido temporal del que se extraeran patrones en la secuencia de *features* en la entrada de la red LSTM.  Se puede usar con `--direction unidirectional` o `--direction bidirectional`.
- `--seq`: Este argumento sirve para especificar el número de features que conformaran la secuencia temporal de entrada a la red LSTM. El parametro depende del número de frames total extraído por video. Los valores posibles para este parámetro son los factores primos del total de frames por video que se establecio en el parametro `--frames` del script `manage_data.py`. Es decir, para un dataset muestrado a 15 frames por video (`python3 manage_data.py --frames 15 ...`) se puede usar `--seq 3`, `--seq 5` o `--seq 15`. 
- `--units`: Este parámetro permite especificar el número de unidades que contendra cada celda de la red LSTM. Se puede usar `--units 128`, `--units 256`, `--units 512`, o `--units 1024`. En realidad, se puede usar cualquier número no hace falta sea potencia de 2.  
- `--data`, `--frames` y `--size`: Sirven para detallar el dataset de *features* alamcenados en disco con el que se va llevar a cabo la operación elegida.