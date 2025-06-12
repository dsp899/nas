import argparse
import os
import ucf101
import cnn 
import config
import tensorflow as tf


def train_cnn(cfg):
    videopaths = ucf101.UCF101(cfg.data['name'], 'split01')
    train_frames = ucf101.Frames(split=videopaths.train_set,config=cfg,num_classes=videopaths.num_classes)
    test_frames = ucf101.Frames(split=videopaths.test_set,config=cfg,num_classes=videopaths.num_classes)
    data = (train_frames.data_train, test_frames.data_test)
    model = cnn.Model(config=cfg,num_classes=videopaths.num_classes)
    model.train(data=data)
 
def eval_cnn(cfg):
    videopaths = ucf101.UCF101(cfg.data['name'], 'split01')
    test_frames = ucf101.Frames(split=videopaths.test_set,config=cfg,num_classes=videopaths.num_classes)
    model = cnn.Model(config=cfg,num_classes=videopaths.num_classes)
    model.evaluate(data=test_frames.data_test)

def predict_cnn(cfg):
    videopaths = ucf101.UCF101(cfg.data['name'], 'split01')
    train_frames = ucf101.Frames(split=videopaths.train_set,config=cfg,num_classes=videopaths.num_classes)
    test_frames = ucf101.Frames(split=videopaths.test_set,config=cfg,num_classes=videopaths.num_classes)
    model = cnn.Model(config=cfg,num_classes=videopaths.num_classes)
    model.predict(mode='train', data=train_frames.data_predict)
    model.predict(mode='test', data=test_frames.data_predict)
    # model.predict(mode='train', data=train_frames.data_predict, total_samples= videopaths.num_videos_train)
    # model.predict(mode='test', data=test_frames.data_predict, total_samples=videopaths.num_videos_test)

def run_main():
    # Argumentos de la línea de comandos
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--operation', type=str, default='', help='{train | eval | predict}')
    ap.add_argument('-a', '--cnn', type=str, default='', help='{vgg16 | resnet50 | inception_v3}')
    ap.add_argument('-d', '--data', type=str, default='all', help='{all | all50 | pmi | pmi50}')
    ap.add_argument('-f', '--frames', type=int, default=15, help='{72 | 36 | 24}')
    ap.add_argument('-s', '--size', type=int, default=299, help='max 299')
    ap.add_argument('-gpu', '--gpu', type=str, default='0', help='0 | 1')

    args = ap.parse_args()

    print('-------------------------------------')
    print('train command line arguments:')
    print(' --operation: ', args.operation)
    print(' --cnn: ', args.cnn)
    print(' --data: ', args.data)
    print(' --frames: ', args.frames)
    print(' --size: ', args.size)
    print(' --gpu: ', args.gpu)
    print('-------------------------------------')
    config.Config.config_device(args.gpu)
    if args.operation == 'train':
        cfg = config.Config(operation=args.operation,cnn=args.cnn,data=args.data,frames=args.frames,size=args.size)
        train_cnn(cfg)

    elif args.operation == 'eval':
        cfg = config.Config(operation=args.operation,cnn=args.cnn,data=args.data,frames=args.frames,size=args.size)
        eval_cnn(cfg)

    elif args.operation == 'predict':
        cfg = config.Config(operation=args.operation,cnn=args.cnn,data=args.data,frames=args.frames,size=args.size)
        predict_cnn(cfg)


if __name__ == '__main__':
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
    print('tf version: ', tf.__version__)
    print('tf.keras version:', tf.keras.__version__)
    tf.keras.utils.set_random_seed(1337)
    run_main()
