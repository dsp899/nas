import argparse
import load_data 
import config
import utils
import numpy as np


def get_data(config_data):
    dataset = load_data.ucf101(name=config_data.name, split='split01')
    data = dataset.get_data(mode=config_data.mode,num_clip_frames=config_data.frames, rescaled_size=config_data.rescaled_size, crop_size=config_data.size)
    #data = dataset.shuffle_data(mode=config_data.mode,data=data)
    """
    if(config_data.mode == 'train'):
        x_train, y_train = data
        np.save('{}/frames_train_{}.npy'.format(data_dirname,dataset_name),x_train)
        np.save('{}/labels_train_{}.npy'.format(data_dirname,dataset_name),y_train)
    elif(config_data.mode == 'test'):
        x_test, y_test = data
        np.save('{}/frames_test_{}.npy'.format(data_dirname,dataset_name),x_test)
        np.save('{}/labels_test_{}.npy'.format(data_dirname,dataset_name),y_test)
    else:
        data_train , data_test = data
        x_train, y_train = data_train
        x_test, y_test = data_test
        np.save('{}/frames_train_{}.npy'.format(data_dirname,dataset_name),x_train)
        np.save('{}/labels_train_{}.npy'.format(data_dirname,dataset_name),y_train)
        np.save('{}/frames_test_{}.npy'.format(data_dirname,dataset_name),x_test)
        np.save('{}/labels_test_{}.npy'.format(data_dirname,dataset_name),y_test)
    """
def run_main():
    
    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--name',
                    type=str,
                    default='',
    	            help='{all | all50 | pmi | pmi50}')
                    
    ap.add_argument('-m', '--mode',
                    type=str,
                    default='all',
    	            help='{all | train | test}')

    ap.add_argument('-f', '--frames',
                    type=str,
                    default='16',
    	            help='{16 | 8 | 4}')

    ap.add_argument('-rs', '--rescaled_size',
                    type=str,
                    default='299',
    	            help='max 299')


    ap.add_argument('-s', '--size',
                    type=str,
                    default='299',
    	            help='max 299')
                    
    args = ap.parse_args()

    print('-------------------------------------')
    print('train command line arguments:')
    print(' --name: ', args.name)
    print(' --mode: ', args.mode)
    print(' --frames: ', args.frames)
    print(' --rescaled_size: ', args.rescaled_size)
    print(' --size: ', args.size)
    print('-------------------------------------')

    config_data = config.ConfigData(dataset_name=args.name,data_mode=args.mode,num_clip_frames=int(args.frames),rescaled_size=int(args.rescaled_size),size=int(args.size))
    get_data(config_data)

if __name__ == '__main__':
    run_main()