import os
import multiprocessing
import tensorflow as tf
batch=16
epochs=10
learning_rate=0.001
predict_frames=36
logical_cores=multiprocessing.cpu_count()

class Config():
    def __init__(self,operation,cnn,data,frames,size):
        self.op = operation
        self.data = {'name':data,'frames':frames, 'size':size} 
        self.cnn = {'name':cnn,'batch':batch,'epochs':epochs,'lr':learning_rate}
        self.predict_frames = predict_frames

    @staticmethod 
    def config_device(gpu_):
        tf.keras.mixed_precision.set_global_policy('mixed_float16') 
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            #os.environ["CUDA_VISIBLE_DEVICES"] = gpu_  # Usar solo la GPU 0 (la primera)
            try:
                # for gpu in gpus:
                tf.config.set_visible_devices(gpus[int(gpu_)], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[int(gpu_)], True)
                # Limitar la memoria a 4 GB (4096 MB)
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpus[0],
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                # )
            except RuntimeError as e:
                print(e)
        else:
            #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.config.threading.set_intra_op_parallelism_threads(logical_cores)
            tf.config.threading.set_inter_op_parallelism_threads(logical_cores)
            print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
            print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())    