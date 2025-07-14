import argparse
import load_data
import load_model_rnn
import config


def train_lstm(config_data,config_model_cnn,config_model_rnn,config_op):
    num_units = int(config_model_rnn.units)
    seq_length = int(config_model_rnn.seq)
    dataset = load_data.ucf101_features(model_architecture=config_model_cnn.model_architecture,model_name=config_model_cnn.model_name)
    data, features_shape = dataset.get_data() # shuffleado por videos
    data_train, data_test = data
    data_train = dataset.create_features_window(data=data_train, seq_length=seq_length)
    data_test = dataset.create_features_window(data=data_test, seq_length=seq_length)
    data = dataset.shuffle_data(mode=config_data.mode, data=(data_train,data_test))
    data_train, data_test = data
    lstm = load_model_rnn.model(model_architecture=config_model_rnn.model_architecture,model_name=config_model_rnn.model_name)
    if config_model_rnn.model == 'single':
        if config_model_rnn.direction == 'unidirectional':
            lstm.create_model_single(num_units=num_units, seq_length=seq_length, num_features=features_shape, num_classes=dataset.num_classes)
        elif config_model_rnn.direction == 'bidirectional':
            lstm.create_model_single_bidirectional(num_units=num_units, seq_length=seq_length, num_features=features_shape, num_classes=dataset.num_classes)
    elif config_model_rnn.model == 'stacked':
        if config_model_rnn.direction == 'unidirectional':
            lstm.create_model_stacked(num_units_layer1=num_units, num_units_layer2= num_units, seq_length=seq_length, num_features=features_shape, num_classes=dataset.num_classes)
        elif config_model_rnn.direction == 'bidirectional':
            lstm.create_model_stacked_bidirectional(num_units_layer1=num_units, num_units_layer2= num_units, seq_length=seq_length, num_features=features_shape, num_classes=dataset.num_classes)
    lstm.add_train_ops(learning_rate=config_op.learning_rate)
    lstm.add_eval_ops()
    lstm.train(epochs=config_op.epochs, batch_size=config_op.batch_size, data_train=data_train, data_test=data_test)


def eval_lstm(config_data,config_model_cnn,config_model_rnn,config_op):
    num_units = int(config_model_rnn.units)
    seq_length = int(config_model_rnn.seq)
    dataset = load_data.ucf101_features(model_architecture=config_model_cnn.model_architecture,model_name=config_model_cnn.model_name)
    data, features_shape = dataset.get_data() # shuffleado por videos
    data_train, data_test = data
    #x_train, y_train = dataset.create_features_window(data=data_train, seq_length=config_data.frames)
    x_test, y_test, videos_id = dataset.create_features_window(data=data_test, seq_length=seq_length)
    lstm = load_model_rnn.model(model_architecture=config_model_rnn.model_architecture,model_name=config_model_rnn.model_name)
    lstm.eval(batch_size=config_op.batch_size, data_test=(x_test,y_test,videos_id))

def run_main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--operation',
                    type=str,
                    default='',
    	            help='{train | eval}')
    
    ap.add_argument('-m', '--model',
        type=str,
        default='single',
    	help='{single | stacked}')
    
    ap.add_argument('-di', '--direction',
        type=str,
        default='unidirectional',
    	help='{unidirectional | bidirectional}')
    
    ap.add_argument('-seq', '--seq',
        type=str,
        default='3',
    	help='{3 | 5 | 15}')
    
    ap.add_argument('-u', '--units',
        type=str,
        default='256',
    	help='{128 | 256 | 512 | 1024}')
                    
    ap.add_argument('-d', '--device',
                    type=str,
                    default='cpu',
    	            help='{gpu | cpu}')
    
    ap.add_argument('-ds', '--data',
                    type=str,
                    default='all',
    	            help='{all | all50 | pmi | pmi50}')
    
    ap.add_argument('-f', '--frames',
                    type=str,
                    default='16',
    	            help='{16 | 8 | 4}')

    ap.add_argument('-s', '--size',
                    type=str,
                    default='299',
    	            help='max 299')                    
                                        
    
    args = ap.parse_args()
    
    print('-------------------------------------')
    print('train command line arguments:')
    print(' --operation: ', args.operation)
    print(' --model: ', args.model)
    print(' --direction: ', args.direction)
    print(' --seq: ', args.seq)
    print(' --units: ', args.units)
    print(' --device: ', args.device)
    print(' --dataset: ', args.data)
    print(' --frames: ', args.frames)
    print(' --size: ', args.size)
    print('-------------------------------------')


    if args.operation == 'train':
        config_data = config.ConfigData(dataset_name= args.data,data_mode='all',num_clip_frames=int(args.frames),size=int(args.size))
        config_model_cnn = config.ConfigModelCNN(model_architecture='vgg16',config_data=config_data)
        config_model_rnn = config.ConfigModelRNN(model=args.model,direction=args.direction,seq=args.seq,units=args.units,model_architecture='vgg16',config_data=config_data)
        config_op = config.ConfigOp(epochs=1, batch_size=15,learning_rate=0.001)
        train_lstm(config_data,config_model_cnn,config_model_rnn,config_op)
    elif args.operation == 'eval':
        config_data = config.ConfigData(dataset_name= args.data,data_mode='test',num_clip_frames=int(args.frames),size=int(args.size))
        config_model_cnn = config.ConfigModelCNN(model_architecture='vgg16',config_data=config_data)
        config_model_rnn = config.ConfigModelRNN(model=args.model,direction=args.direction,seq=args.seq,units=args.units,model_architecture='vgg16',config_data=config_data)
        config_op = config.ConfigOp(batch_size=1)
        eval_lstm(config_data,config_model_cnn,config_model_rnn,config_op)


if __name__ == '__main__':
    run_main()

