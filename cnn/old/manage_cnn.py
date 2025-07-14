import argparse
import load_data
import load_model_cnn as lm
import config
import tensorflow as tf

def train_cnn(config_data,config_model,config_op):
    dataset = load_data.ucf101_frames(name=config_data.name,frames=config_data.frames,size=config_data.size)
    data = dataset.get_data(mode=config_data.mode)
    data = dataset.shuffle_data(mode=config_data.mode, data=data)
    cnn = lm.model(load_mode=config_model.load_mode,model_architecture=config_model.model_architecture,model_name=config_model.model_name)
    cnn.create_backend(device=config_model.device,size=config_data.size)
    cnn.add_frontend_ops_default(num_classes=dataset.num_classes)
    cnn.tunning(num_evals=5,epochs=config_op.epochs,batch_size=config_op.batch_size,learning_rate=config_op.learning_rate,data=data)
    cnn.save()

def eval_cnn(config_data,config_model,config_op):
    dataset = load_data.ucf101_frames(name=config_data.name,frames=config_data.frames,size=config_data.size)
    data = dataset.get_data(mode=config_data.mode)
    data = dataset.shuffle_data(mode=config_data.mode, data=data)
    cnn = lm.model(load_mode=config_model.load_mode,model_architecture=config_model.model_architecture,model_name=config_model.model_name)
    cnn.create_backend(device=config_model.device,size=config_data.size)
    cnn.add_frontend_ops_default(num_classes=dataset.num_classes)
    cnn.eval(batch_size=config_op.batch_size, data=data)

def infer_cnn(config_data,config_model,config_op):
    dataset = load_data.ucf101_frames(name=config_data.name,frames=config_data.frames,size=config_data.size)
    data = dataset.get_data(mode=config_data.mode)
    #data = dataset.shuffle_data(mode=config_data.mode, data=data)
    cnn = lm.model(load_mode=config_model.load_mode,model_architecture=config_model.model_architecture,model_name=config_model.model_name)
    cnn.create_backend(device=config_model.device,size=config_data.size)
    cnn.add_frontend_ops_default(num_classes=dataset.num_classes)
    cnn.infer(batch_size=config_op.batch_size, mode=config_data.mode, data=data)

def freeze_cnn(config_data,config_model):    
    cnn = lm.model(load_mode=config_model.load_mode,model_architecture=config_model.model_architecture,model_name=config_model.model_name)
    cnn.create_backend(device=config_model.device,size=config_data.size)
    cnn.freeze()



def run_main():
    
    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--operation',
                    type=str,
                    default='',
    	            help='{train | eval | infer | freeze}')

    ap.add_argument('-m', '--model',
                    type=str,
                    default='',
    	            help='{vgg16 | resnet50 | inception_v3}')                
                    
    ap.add_argument('-d', '--device',
                    type=str,
                    default='cpu',
    	            help='{gpu | cpu}')
    
    ap.add_argument('-i', '--infer',
                    type=str,
                    default='',
    	            help='{train | test}')

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
    print(' --device: ', args.device)
    print(' --infer: ', args.infer)
    print(' --dataset: ', args.data)
    print(' --frames: ', args.frames)
    print(' --size: ', args.size)    
    print('-------------------------------------')

    if args.operation == 'train':
        config_data = config.ConfigData(dataset_name= args.data,data_mode='all',num_clip_frames=int(args.frames),size=int(args.size))
        config_model = config.ConfigModelCNN(load_mode='kerasModel',model_architecture=args.model,device=args.device,config_data=config_data)
        config_op = config.ConfigOp(epochs=1,batch_size=8,learning_rate=0.001)
        train_cnn(config_data,config_model,config_op)
    elif args.operation == 'eval':
        config_data = config.ConfigData(dataset_name= args.data,data_mode='test',num_clip_frames=int(args.frames),size=int(args.size))
        config_model = config.ConfigModelCNN(load_mode='savedModel',model_architecture=args.model,device=args.device,config_data=config_data)
        config_op= config.ConfigOp(batch_size=1)
        eval_cnn(config_data,config_model,config_op)
    elif args.operation == 'infer':
        config_data = config.ConfigData(dataset_name= args.data,data_mode=args.infer,num_clip_frames=int(args.frames),size=int(args.size))
        config_model = config.ConfigModelCNN(load_mode='savedModel',model_architecture=args.model,device=args.device,config_data=config_data)
        config_op= config.ConfigOp(batch_size=1)
        infer_cnn(config_data,config_model,config_op)
    elif args.operation == 'freeze':
        config_data = config.ConfigData(dataset_name= args.data,num_clip_frames=int(args.frames),size=int(args.size))
        config_model = config.ConfigModelCNN(load_mode='savedModel',model_architecture=args.model,device=args.device,config_data=config_data)
        freeze_cnn(config_data,config_model)


if __name__ == '__main__':
    print('tf version: ', tf.__version__)
    print('tf.keras version:', tf.keras.__version__)
    tf.compat.v1.set_random_seed(264)
    #tf.keras.utils.set_random_seed(1337)
    run_main()
