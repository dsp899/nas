import os
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import random

class Features():
    
    def __init__(self,mode,config):
        self.mode = mode
        self.config = config
        self.features_tag = f"{self.config.data['cnn']}_{self.config.data['name']}_{self.config.data['frames']}_frames_{self.config.data['size']}_size"
        self.sequences_tag = f"{self.config.data['cnn']}_{self.config.data['name']}_{self.config.data['frames']}_frames_{self.config.data['size']}_size_{self.config.data['seq']}_seq_{self.config.data['state']}"
        self.features_dir = f"{os.path.abspath(os.path.curdir)}/data/features/{self.config.data['name']}/{self.config.data['cnn']}/{self.features_tag}"
        self.sequences_dir = f"{os.path.abspath(os.path.curdir)}/data/sequences/{self.config.data['name']}/{self.config.data['cnn']}/{self.sequences_tag}"
        self.load()
        self.generate_sequences()
        self.save()

    def load(self):
        # videopaths = UCF101(self.config.data['name'], 'split01')
        # CNN_FEATURE_SHAPE = {
        #     "vgg16": 512,
        #     "resnet50": 2048,
        #     "inceptionV3": 2048,  # Ajusta si InceptionV3 usa otro tamaño
        # }
        # def get_feature_shape(filepath):
        #     filepath_lower = filepath.lower()  # Para hacer la búsqueda insensible a mayúsculas
            
        #     for cnn_name in CNN_FEATURE_SHAPE:
        #         if cnn_name in filepath_lower:
        #             return CNN_FEATURE_SHAPE[cnn_name]
            
        #     # Si no se encuentra ninguna red, lanza un error o devuelve un valor por defecto
        #     raise ValueError(
        #         f"No se pudo detectar la red (vgg16/resnet50/inceptionV3) en el filepath: {filepath}"
        #     )
        # num_videos = videopaths.num_videos_train if self.mode == 'train' else videopaths.num_videos_test
        # self.videos = np.memmap(f"{self.features_dir}/features_{self.mode}_{self.features_tag}.npy",dtype=np.float32,mode='r',shape=(num_videos, self.config.data['frames'], get_feature_shape(f"{self.features_dir}/features_{self.mode}_{self.features_tag}.npy")))
        # self.labels = np.memmap(f"{self.features_dir}/labels_{self.mode}_{self.features_tag}.npy",dtype=np.float32,mode='r',shape=(num_videos, videopaths.num_classes))
        # self.video_id = np.memmap(f"{self.features_dir}/video_id_{self.mode}_{self.features_tag}.npy",dtype=np.float32,mode='r',shape=(num_videos))   
        self.videos = np.load(f"{self.features_dir}/features_{self.mode}_{self.features_tag}.npy")
        self.labels = np.load(f"{self.features_dir}/labels_{self.mode}_{self.features_tag}.npy")
        self.video_id = np.load(f"{self.features_dir}/video_id_{self.mode}_{self.features_tag}.npy")
        print(f"Load {self.mode} features shape: {self.videos.shape}")
        print(f"Load {self.mode} labels shape: {self.labels.shape}")
        print(f"Load {self.mode} videos_id shape: {self.video_id.shape}")
    
    
    def generate_sliding_windows(self, sequence, window_size, state):
        stride = 3 if self.config.data['state'] == "stateless" else self.config.data['seq']
        stride = self.config.data['seq'] if self.mode == 'test' else stride
        sequence = np.array(sequence)
        N = len(sequence)
        windows = []

        for start in range(0, N - window_size + 1, stride):
            window = sequence[start:start + window_size]
            windows.append(window)

        return np.stack(windows)
        
    def generate_sequences(self):
        self.features_seqs = []
        for video in self.videos:
            video_windows = self.generate_sliding_windows(video, window_size=self.config.data['seq'], state=self.config.data['state'])
            self.features_seqs.append(video_windows)
        
        self.features_seqs = np.stack(self.features_seqs)
        features_size = self.features_seqs.nbytes / (1024**2)
        print(f"Save features shape: {self.features_seqs.shape}, Features size: {features_size:.2f} MB")

    def save(self):
        os.makedirs(self.sequences_dir, exist_ok=True)
        # np.save(f"{self.sequences_dir}/sequences_{self.mode}_{self.sequences_tag}.npy",self.features_seqs)
        # np.save(f"{self.sequences_dir}/labels_{self.mode}_{self.sequences_tag}.npy",self.labels)
        # np.save(f"{self.sequences_dir}/videos_id_{self.mode}_{self.sequences_tag}.npy",self.video_id)
        np.save(f"{self.sequences_dir}/sequences_{self.mode}_{self.sequences_tag}.npy",self.features_seqs)
        np.save(f"{self.sequences_dir}/labels_{self.mode}_{self.sequences_tag}.npy",self.labels[:,0,:])
        np.save(f"{self.sequences_dir}/videos_id_{self.mode}_{self.sequences_tag}.npy",self.video_id[:,0])

class Sequences():
    def __init__(self,mode,config):
        self.mode = mode
        self.config = config
        self.sequences_tag = f"{self.config.data['cnn']}_{self.config.data['name']}_{self.config.data['frames']}_frames_{self.config.data['size']}_size_{self.config.data['seq']}_seq_{self.config.data['state']}"
        self.sequences_dir = f"{os.path.abspath(os.path.curdir)}/data/sequences/{self.config.data['name']}/{self.config.data['cnn']}/{self.sequences_tag}"
        self.check()
        self.load()
        self.generate()

    def check(self):

        if os.path.exists(self.sequences_dir):
            if os.path.isdir(self.sequences_dir):
                print("El directorio existe.")
            else:
                print("Existe, pero no es un directorio.")
        else:
            print("El directorio no existe.")
            Features(mode='train',config=self.config)
            Features(mode='test',config=self.config)

    def load(self):
        self.videos = np.load(f"{self.sequences_dir}/sequences_{self.mode}_{self.sequences_tag}.npy")
        self.labels = np.load(f"{self.sequences_dir}/labels_{self.mode}_{self.sequences_tag}.npy")
        self.videos_id = np.load(f"{self.sequences_dir}/videos_id_{self.mode}_{self.sequences_tag}.npy")
        self.utils()
        if self.config.data['state'] == 'stateless':
            self.videos = self.videos.reshape(self.num_videos * self.num_sequences, self.sequence_size, self.num_features)
            self.labels = np.repeat(self.labels, repeats=self.num_sequences, axis=0)
            self.videos_id = np.repeat(self.videos_id, repeats=self.num_sequences, axis=0)

        features_size = self.videos.nbytes / (1024**2)
        print(f"Load {self.mode} sequences shape: {self.videos.shape}, Sequences size: {features_size:.2f} MB")
        print(f"Load {self.mode} labels shape: {self.labels.shape}")
        print(f"Load {self.mode} videos_id shape: {self.videos_id.shape}")

    def utils(self):
        self.num_videos = self.videos.shape[0]
        self.num_sequences =  self.videos.shape[1]
        self.sequence_size = self.videos.shape[2]
        self.num_features = self.videos.shape[3]
        self.num_classes = self.labels.shape[-1]

    def read_data(self):
        for sequence, label, video_id in zip(self.videos,self.labels,self.videos_id):
            # print(f"features_seqs shape: {sequences.shape}")
            # print(f"labels_seqs shape: {label.shape}")
            # print(f"vid_seqs shape: {video_id.shape}")
            yield sequence, label, video_id

    def generate(self):
        output_signature_statefull = (tf.TensorSpec(shape = (self.num_sequences,self.sequence_size, self.num_features), dtype = tf.float32),tf.TensorSpec(shape = (self.num_classes), dtype = tf.int16),tf.TensorSpec(shape = (), dtype = tf.int16))
        output_signature_stateless = (tf.TensorSpec(shape = (self.sequence_size, self.num_features), dtype = tf.float32),tf.TensorSpec(shape = (self.num_classes), dtype = tf.int16),tf.TensorSpec(shape = (), dtype = tf.int16))
        self.data = tf.data.Dataset.from_generator(self.read_data, output_signature = output_signature_statefull if self.config.data['state'] == "statefull" else output_signature_stateless)
        self.data = self.data.map(self.filter_output_train) if self.config.data['state'] == "statefull" else self.data
        self.data_train = self.data.shuffle(5000).prefetch(buffer_size = tf.data.AUTOTUNE).batch(self.config.params['batch']) 
        self.data_test = self.data.prefetch(buffer_size = tf.data.AUTOTUNE).batch(self.config.params['batch'])
        

    def filter_output_train(self, frame, label, video_id):
        return frame, label  # Excluir video_id
        

if __name__ == '__main__':
    print("__main__")