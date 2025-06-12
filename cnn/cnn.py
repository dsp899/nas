import os
import json
import numpy as np
import tensorflow as tf

class SaveProgress(tf.keras.callbacks.Callback):
    def __init__(self, model, batch_size, lr, current_epoch, best_val_loss, best_val_acc, info_path, best_model_path, last_model_path):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.current_epoch = current_epoch
        self.best_val_loss = best_val_loss
        self.best_val_acc = best_val_acc
        self.info_path = info_path
        self.best_model_path = best_model_path
        self.last_model_path = last_model_path

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        current_val_loss = logs.get('val_loss')
        current_val_acc = logs.get('val_accuracy')

        if self.best_val_loss == 0 or current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        if self.best_val_acc == 0 or current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.model.save(self.best_model_path)
        os.makedirs(os.path.dirname(self.info_path), exist_ok=True)
        with open(self.info_path, "w") as f:
            progress = {"best_val_loss": self.best_val_loss,
                        "best_val_acc": self.best_val_acc,
                        "epoch": self.current_epoch,
                        "last_val_loss": current_val_loss,
                        "last_val_acc": current_val_acc,
                        "batch_size": self.batch_size,
                        "lr": self.lr}
            json.dump(progress, f)
        self.model.save(self.last_model_path)


class Model():

    def __init__(self, config, num_classes):
        self.model_filename = f"{config.cnn['name']}_{config.data['name']}_{config.data['frames']}_frames_{config.data['size']}_size"
        self.predict_filename = f"{config.cnn['name']}_{config.data['name']}_{config.predict_frames}_frames_{config.data['size']}_size"
        self.models_path = f"{os.path.abspath(os.path.curdir)}/models/keras/{config.data['name']}/cnn/{config.cnn['name']}/{self.model_filename}"
        self.data_path = f"{os.path.abspath(os.path.curdir)}/data/features/{config.data['name']}/{config.cnn['name']}/{self.predict_filename}"
        self.info_path = f"{self.models_path}/{self.model_filename}_info.json"
        self.best_model_path = f"{self.models_path}/{self.model_filename}_best.h5" 
        self.last_model_path = f"{self.models_path}/{self.model_filename}_last.h5"
        self.config = config
        self.num_classes = num_classes
        self.create_model()
        

    def load_train(self): 
        if os.path.exists(self.info_path):
            with open(self.info_path, "r") as f:
                info = json.load(f)
                self.current_epoch = info.get("epoch", 0)
                self.best_val_loss = info.get("best_val_loss", 0)
                self.best_val_acc = info.get("best_val_acc", 0)
                self.current_epoch = 0 if self.current_epoch >= self.config.cnn['epochs'] else self.current_epoch
        else:
            self.current_epoch = 0
            self.best_val_loss = 0
            self.best_val_acc = 0
        print(self.current_epoch)
        if self.current_epoch != 0:
            self.model = tf.keras.models.load_model(self.last_model_path, compile=True)
        else:     
            if self.config.cnn['name'] == 'vgg16':
                base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(self.config.data['size'], self.config.data['size'], 3))
            elif self.config.cnn['name']  == 'resnet50':
                base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(self.config.data['size'], self.config.data['size'], 3))
            elif self.config.cnn['name']  == 'inceptionV3':
                base_model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(self.config.data['size'], self.config.data['size'], 3))
            else:
                raise ValueError("Invalid model architecture")

            # Adding custom layers on top of the base model
            pooling = tf.keras.layers.GlobalAveragePooling2D(name='feature_extractor')(base_model.output)
            dropout = tf.keras.layers.Dropout(0.5)(pooling)   
            dense1 = tf.keras.layers.Dense(2048, activation='relu')(pooling)  
            dropout = tf.keras.layers.Dropout(0.5)(dense1) 
            output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dropout)  
            
            self.model = tf.keras.Model(inputs=base_model.input, outputs=output)
            self.model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=self.config.cnn['lr']),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=['accuracy'])
            self.model.summary()
    
            self.feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=pooling)

    def load_eval(self):
        self.model = tf.keras.models.load_model(self.best_model_path)

    def load_predict(self):
        self.model = tf.keras.models.load_model(self.best_model_path)
        features = self.model.get_layer(name="feature_extractor").output
        print(features)
        self.extractor = tf.keras.Model(inputs=self.model.input, outputs=features)


    def create_model(self):
        if self.config.op == 'train':
            self.load_train()
        elif self.config.op == 'eval':
            self.load_eval()
        elif self.config.op == 'predict':
            self.load_predict()
        else:
            raise ValueError('Unsupported load mode')

    def train(self, data):
        data_train, data_test = data
        save_progress = SaveProgress(self.model,self.config.cnn['batch'],self.config.cnn['lr'],self.current_epoch,self.best_val_loss,self.best_val_acc,self.info_path,self.best_model_path, self.last_model_path)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.model.fit(
            data_train, 
            validation_data=data_test,
            initial_epoch=self.current_epoch, 
            epochs=self.config.cnn['epochs'],
            callbacks=[early_stop, save_progress]
        )

    def evaluate(self, data):
        test_loss, test_acc = self.model.evaluate(data)
        print(f'Test accuracy: {test_acc * 100:.2f}%, Test loss: {test_loss:.2f}')

    def count_dataset(self, dataset):
        count = 0
        for _ in dataset:
            count += 1
        return count


    # def predict(self, mode, data, total_samples):
    #     frames_size = 0
    #     features_size = 0

    #     os.makedirs(self.data_path, exist_ok=True)

    #     sample_frame, sample_label, sample_video_id = next(iter(data))
    #     sample_feature = self.extractor(sample_frame)
    #     feature_shape = sample_feature.shape[1:]  # sin batch

    #     features_array = np.memmap(f'{self.data_path}/features_{mode}_{self.predict_filename}.npy',
    #                             dtype=np.float32, mode='w+', shape=(total_samples, *sample_feature.shape))
    #     labels_array = np.memmap(f'{self.data_path}/labels_{mode}_{self.predict_filename}.npy',
    #                             dtype=sample_label.numpy().dtype, mode='w+', shape=(total_samples, *sample_label.shape))
    #     video_id_array = np.memmap(f'{self.data_path}/video_id_{mode}_{self.predict_filename}.npy',
    #                             dtype=sample_video_id.numpy().dtype, mode='w+', shape=(total_samples, *sample_video_id.shape))

    #     idx = 0
    #     for frame_batch, label_batch, video_id_batch in data:
    #         batch_size = frame_batch.shape[0]
    #         frames_size += (frame_batch.numpy().nbytes / (1024**2))

    #         features_batch = self.extractor(frame_batch)
    #         features_np = features_batch.numpy()

    #         features_array[idx:idx+batch_size] = features_np
    #         labels_array[idx:idx+batch_size] = label_batch.numpy()
    #         video_id_array[idx:idx+batch_size] = video_id_batch.numpy()

    #         idx += batch_size

    #     features_array.flush()
    #     labels_array.flush()
    #     video_id_array.flush()

    #     features_size = features_array.nbytes / (1024**2)

    #     print(f"Shapes features: {features_array.shape}, labels: {labels_array.shape}, video ids: {video_id_array.shape}")
    #     print(f"Frames size: {frames_size:.2f} MB, Features size: {features_size:.2f} MB")




    
    def predict(self, mode, data):
        frames_size = 0
        features_size = 0
        features_list = []
        labels_list = []
        video_id_list = []
        for frame, label, video_id in data:
            frames_size += (frame.numpy().nbytes / (1024**2))
            #print(video_id)
            feature = self.extractor(frame)
            features_list.append(feature.numpy().squeeze())  # Convertir a numpy
            labels_list.append(label.numpy())
            video_id_list.append(video_id.numpy())
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        video_id_array = np.array(video_id_list)
        print(f"Shapes features: {features_array.shape}, labels:{labels_array.shape}, video ids: {video_id_array.shape}")
        features_size = features_array.nbytes / (1024**2)
        print(f"Frames size: {frames_size:.2f} MB, Features size: {features_size:.2f} MB")
        os.makedirs(f'{self.data_path}', exist_ok=True)
        np.save(f'{self.data_path}/features_{mode}_{self.predict_filename}.npy',features_array)
        np.save(f'{self.data_path}/labels_{mode}_{self.predict_filename}.npy', labels_array)
        np.save(f'{self.data_path}/video_id_{mode}_{self.predict_filename}.npy', video_id_array)