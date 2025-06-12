import argparse
import os
import ucf101
import rnn
import config
import tensorflow as tf

import psutil
import gc

from config import print_memory

# def train_lstm(cfg):
#     train_sequences = ucf101.Sequences(mode='train',config=cfg)
#     test_sequences = ucf101.Sequences(mode='test',config=cfg)
#     data =(train_sequences.data_train, test_sequences.data_test)
#     model = rnn.ModelTrainer(config=cfg,num_sequences=train_sequences.num_sequences,num_features=train_sequences.num_features,num_classes=train_sequences.num_classes)
#     test_acc = model.train(data=data)
#     return test_acc

def train_lstm(cfg):
    print("\n" + "="*50)
    print("Iniciando proceso de entrenamiento LSTM")
    print("="*50 + "\n")
    
    print_memory()  # Antes
    
    print("[Paso 1/5] Cargando secuencias de entrenamiento...")
    train_sequences = ucf101.Sequences(mode='train', config=cfg)
    print("✓ Secuencias de entrenamiento cargadas correctamente")
    print("   - Información de características disponible")
    print("   - Clases identificadas\n")
    print_memory()  # Antes

    print_memory()
    print("[Paso 2/5] Cargando secuencias de validación...")
    test_sequences = ucf101.Sequences(mode='test', config=cfg)
    print("✓ Secuencias de validación cargadas correctamente\n")
    print_memory()  # Antes
    print("[Paso 3/5] Preparando conjuntos de datos...")
    data = (train_sequences.data_train, test_sequences.data_test)
    print("✓ Datos organizados correctamente")
    print("   - Estructura de entrenamiento lista")
    print("   - Estructura de prueba lista\n")
    print("IDs activos:", tf.keras.backend.get_uid())
    print_memory()  # Antes
    print("[Paso 4/5] Construyendo modelo...")
    model = rnn.ModelTrainer(
        config=cfg,
        num_sequences=train_sequences.num_sequences,
        num_features=train_sequences.num_features,
        num_classes=train_sequences.num_classes
    )
    print("✓ Arquitectura del modelo creada exitosamente\n")
    print("IDs activos:", tf.keras.backend.get_uid())
    print_memory()  # Antes
    print("[Paso 5/5] Iniciando fase de entrenamiento...")
    print("-"*40)
    print("Procesando iteraciones...")
    test_acc = model.train(data=data)
    print("-"*40)
    print("\nProceso de entrenamiento finalizado")
    print("Métrica de evaluación obtenida")
    
    print("\n" + "="*50)
    print("Pipeline de entrenamiento completado")
    print("="*50 + "\n")
    print("IDs activos:", tf.keras.backend.get_uid())
    print_memory()  # Antes
    tf.keras.backend.clear_session()
    print_memory()  # Antes
    train_sequences.videos = None
    train_sequences.labels = None
    train_sequences.videos_id = None
    train_sequences.data = None
    train_sequences.data_train = None
    train_sequences.data_test = None
    test_sequences.videos = None
    test_sequences.labels = None
    test_sequences.videos_id = None
    test_sequences.data = None
    test_sequences.data_train = None
    test_sequences.data_test = None
    model.model = None
    model.loss_fn = None
    model.optimizer = None

    del model, train_sequences, test_sequences, data
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    print_memory() 
    print("IDs activos:", tf.keras.backend.get_uid())
    return test_acc


def eval_lstm(cfg):
    test_sequences = ucf101.Sequences(mode='test',config=cfg)
    model = rnn.Model(config=cfg,num_sequences=train_sequences.num_sequences,num_features=train_sequences.num_features,num_classes=train_sequences.num_classes)
    metrics = model.evaluate(data=test_sequences.data_test)
    return metrics['acc']



def parse_tuple(s):
    try:
        return tuple(map(int, s.strip("() ").split(",")))
    except Exception:
        raise argparse.ArgumentTypeError(f"Formato inválido: {s}")

def run_main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--operation',
                    type=str,
                    default='',
    	            help='{train | eval}')

    ap.add_argument('-t', '--rnn',
                    type=str,
                    default='lstm',
                    help='{lstm | gru}')
    
    ap.add_argument('-direction', '--direction',
                    type=str,
                    default='unidirectional',
                    help='{unidirectional | bidirectional}')
    
    ap.add_argument('-u', '--units',
                    type=parse_tuple, 
                    help='{(128,128,128)}')
                    
    ap.add_argument('-c', '--cnn',
                    type=str,
                    default='vgg16',
    	            help='{vgg16 | resnet50 | inceptionV3}')
    
    ap.add_argument('-d', '--data',
                    type=str,
                    default='all',
    	            help='{all | all50 | pmi | pmi50}')
    
    ap.add_argument('-f', '--frames',
                    type=int,
                    default=15,
    	            help='{15 | 10 | 5}')

    ap.add_argument('-s', '--size',
                    type=int,
                    default=299,
    	            help='max 299')

    ap.add_argument('-seq', '--seq',
                    type=int,
                    default=3,
                    help='{3 | 5 | 15}')

    ap.add_argument('-state', '--state',
                    type=str,
                    default='stateless',
                    help='{stateless | statefull}')
    ap.add_argument('-gpu', '--gpu', 
                    type=str,
                    default='0',
                    help='0 | 1')
                        
    args = ap.parse_args()
    
    print('-------------------------------------')
    print('train command line arguments:')
    print(' --operation: ', args.operation)
    print(' --rnn: ', args.rnn)
    print(' --direction: ', args.direction)
    print(' --units: ', args.units)
    print(' --cnn: ', args.cnn)
    print(' --data: ', args.data)
    print(' --frames: ', args.frames)
    print(' --size: ', args.size)
    print(' --seq: ', args.seq)
    print(' --state: ', args.state)
    print(' --gpu: ', args.gpu)
    print('-------------------------------------')

    config.Config.config_device(args.gpu)
    if args.operation == 'train':
        cfg = config.Config(operation=args.operation,rnn=args.rnn,direction=args.direction,units=args.units,cnn=args.cnn,data=args.data,frames=args.frames,size=args.size,seq=args.seq,state=args.state)
        train_lstm(cfg)
    elif args.operation == 'eval':
        cfg = config.Config(operation=args.operation,rnn=args.rnn,direction=args.direction,units=args.units,cnn=args.cnn,data=args.data,frames=args.frames,size=args.size,seq=args.seq,state=args.state)
        eval_lstm(cfg)


if __name__ == '__main__':
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
    print('tf version: ', tf.__version__)
    print('tf.keras version:', tf.keras.__version__)
    #tf.config.experimental.enable_op_determinism
    tf.keras.utils.set_random_seed(1337)
    run_main()

