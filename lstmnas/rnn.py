# import os
# import numpy as np
# import tensorflow as tf
# import time
# from tqdm import tqdm
# from collections import Counter
# from typing import Dict, List, Tuple

# class VideoEvaluator:
#     """Clase optimizada para evaluación de videos con estrategia seleccionada"""

#     @staticmethod
#     @tf.function
#     def evaluate_video(all_logits: np.ndarray,
#                       strategy: str) -> np.ndarray:
#         if strategy == 'average':
#             return tf.reduce_mean(all_logits, axis=0)
#         elif strategy == 'majority':
#             return VideoEvaluator._majority_vote(all_logits)
#         elif strategy == 'max_prob':
#             return VideoEvaluator._max_probability(all_logits)
#         elif strategy == 'temporal':
#             return VideoEvaluator._temporal_attention(all_logits)
#         else:
#             return tf.reduce_mean(all_logits, axis=0)  # Default

#     @staticmethod
#     @tf.function
#     def _majority_vote(logits: tf.Tensor) -> tf.Tensor:
#         predictions = tf.argmax(logits, axis=1)
#         predictions = tf.cast(predictions, tf.int32)
#         majority = tf.argmax(tf.math.bincount(predictions))
#         return tf.one_hot(majority, logits.shape[1])

#     @staticmethod
#     @tf.function
#     def _max_probability(logits: tf.Tensor) -> tf.Tensor:
#         probs = tf.nn.softmax(logits)
#         max_prob_index = tf.argmax(tf.reduce_max(probs, axis=1))
#         return logits[max_prob_index]

#     @staticmethod
#     @tf.function
#     def _temporal_attention(logits: tf.Tensor) -> tf.Tensor:
#         weights = tf.linspace(0.1, 1.0, tf.shape(logits)[0])
#         return tf.reduce_sum(logits * weights[:, tf.newaxis], axis=0) / tf.reduce_sum(weights)

# class TrainingUtils:
#     """Clase con utilidades comunes para ambos tipos de entrenamiento"""
    
#     @staticmethod
#     def init_metrics() -> Tuple[Dict, Dict]:
#         """Inicializa métricas comunes para train y test"""
#         train_metrics = {
#             'loss': tf.keras.metrics.Mean(),
#             'acc': tf.keras.metrics.CategoricalAccuracy(),
#             'current_batch': 0,
#             'batch_loss': 0.0,
#             'batch_acc': 0.0
#         }
        
#         test_metrics = {
#             'loss': tf.keras.metrics.Mean(),
#             'acc': tf.keras.metrics.CategoricalAccuracy(),
#             'current_batch': 0
#         }
        
#         return train_metrics, test_metrics
    
#     @staticmethod
#     def init_history() -> Dict:
#         """Inicializa el historial de entrenamiento"""
#         return {
#             'epoch': [],
#             'train_loss': [],
#             'train_acc': [],
#             'test_loss': [],
#             'test_acc': [],
#             'time_per_epoch': [],
#             'best_test_acc': 0.0
#         }

#     @staticmethod
#     def _update_history(history: Dict, train_metrics: Dict, 
#                        test_metrics: Dict, epoch: int, epoch_time: float):
#         """Actualiza el historial con los resultados de la época"""
#         history['epoch'].append(epoch + 1)
#         history['train_loss'].append(train_metrics['loss'].result().numpy())
#         history['train_acc'].append(train_metrics['acc'].result().numpy())
#         history['test_loss'].append(test_metrics['loss'].result().numpy())
#         history['test_acc'].append(test_metrics['acc'].result().numpy())
#         history['time_per_epoch'].append(epoch_time)
        
#         # Logging detallado
#         print(f"\nEpoch {epoch+1} - "
#               f"Train Loss: {history['train_loss'][-1]:.4f}, "
#               f"Train Acc: {history['train_acc'][-1]:.2%} - "
#               f"Test Loss: {history['test_loss'][-1]:.4f}, "
#               f"Test Acc: {history['test_acc'][-1]:.2%} - "
#               f"Time: {epoch_time:.1f}s")
    
#     @staticmethod
#     def update_progress_bar(progress_bar: tqdm, metrics: Dict, stage: str):
#         """Actualiza la barra de progreso con métricas"""
#         if stage == 'train':
#             progress_bar.set_postfix({
#                 'loss': f"{metrics['batch_loss']:.4f}",
#                 'acc': f"{metrics['batch_acc']:.2%}",
#                 'avg_loss': f"{metrics['loss'].result():.4f}",
#                 'avg_acc': f"{metrics['acc'].result():.2%}"
#             })
#         else:
#             progress_bar.set_postfix({
#                 'loss': f"{metrics['loss'].result():.4f}",
#                 'acc': f"{metrics['acc'].result():.2%}"
#             })


# # Integración con tu clase Model existente
# class Model():
    
#     def __init__(self, config, num_sequences, num_features, num_classes):
#         # ... (tu código existente)
#         self.config = config
#         self.num_sequences = num_sequences
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.strategy = 'majority'
#         self.sequences_tag = f"{self.config.data['cnn']}_{self.config.data['name']}_{self.config.data['frames']}_frames_{self.config.data['size']}_size_{self.config.data['seq']}_seq_{self.config.data['state']}"
#         self.generate_model_tag()
#         self.models_dir = f"{os.path.abspath(os.path.curdir)}/models/keras/{self.config.data['name']}/rnn/{self.config.data['cnn']}/{self.sequences_tag}/{self.model_tag}"
#         self.load_model()
    
    
#     def generate_model_tag(self):
#         self.model_tag = f"{self.config.arch['type']}"
#         for unit in self.config.arch['units']:
#             self.model_tag += f"_{unit}u"
#         self.model_tag += f"_{self.config.arch['direction']}"

#     def load_model(self):
#         if self.config.op == 'train':
#             self.load_train()
#         elif self.config.op == 'eval':
#             self.load_eval()
#         else:
#             print("Error")

#     def load_train(self):
#         # recurrent_dropout=0.3
#         # dense_dropout=0.2
#         # dense_units=64
#         recurrent_dropout=0
#         dense_dropout=0.5
#         dense_units=256  
#         l2_reg=0.01

#         units = tuple(u for u in self.config.arch['units'] if u not in {0})
  
#         batch_size = self.config.params['batch'] if self.config.data['state'] == 'statefull' else None

#         # Definición de la forma de entrada
#         if self.config.data['state'] == 'stateless':
#             input_shape = (self.config.data['seq'], self.num_features)
#             inputs = tf.keras.Input(shape=input_shape, batch_size=None)  # Sin batch size fijo
#         else:  # Stateful
#             # Para stateful, debemos especificar el batch_size en el Input
#             input_shape = (self.config.data['seq'], self.num_features)
#             inputs = tf.keras.Input(batch_input_shape=(batch_size, *input_shape))  # batch_size incluido

#         RNNLayer = tf.keras.layers.LSTM if self.config.arch['type'] == 'lstm' else tf.keras.layers.GRU
#         state = True if self.config.data['state'] == 'statefull' else False

#         x = inputs

#         for i, unit in enumerate(units):
#             return_sequences = i < len(units) - 1  # True excepto para la última capa
            
#             recurrent_layer = RNNLayer(units=unit,return_sequences=return_sequences,stateful=state,recurrent_dropout=recurrent_dropout)
#             # print(unit)
#             if self.config.arch['direction'] == 'bidirectional':
#                 recurrent_layer = tf.keras.layers.Bidirectional(recurrent_layer)
            
#             x = recurrent_layer(x)
            
#             x = tf.keras.layers.BatchNormalization()(x)
        
#         x = tf.keras.layers.Dropout(dense_dropout)(x)
        
#         x = tf.keras.layers.Dense(dense_units,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    
#         outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        
#         self.model = tf.keras.Model(inputs=inputs, outputs=outputs)


#         self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
#         self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.config.params['lr'],clipvalue=1.0)
#         # self.modelcompile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=self.config.params['lr'],clipvalue=1.0),       #Adagrad(learning_rate=learning_rate),
#         #                     loss=tf.keras.losses.CategoricalCrossentropy(),
#         #                     metrics=['accuracy'])
#         self.model.summary()
    
#     def load_eval(self):
#         self.model = tf.keras.models.load_model(f'{self.models_dir}/{self.model_tag}.h5')

# class ModelTrainer(Model):
#     """Clase que contiene los métodos de entrenamiento optimizados"""
    
#     def __init__(self, config, num_sequences, num_features, num_classes):
#         super().__init__(config, num_sequences, num_features, num_classes)
    
#     @tf.function(experimental_relax_shapes=True)
#     def train_stateless_step(self, batch, labels):
#         """Paso de entrenamiento optimizado con tf.function"""
#         with tf.GradientTape() as tape:
#             logits = self.model(batch, training=True)
#             loss = self.loss_fn(labels, logits)
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#         return loss, logits
    
#     @tf.function(experimental_relax_shapes=True)
#     def test_stateless_step(self, batch):
#         """Paso de evaluación optimizado con tf.function"""
#         return self.model(batch, training=False)

#     # Compilar funciones críticas
#     @tf.function
#     def train_statefull_step(self, x_batch, y_batch):
#         self.model.reset_states()
#         with tf.GradientTape() as tape:
#             for t in range(self.num_sequences):
#                 subseq = x_batch[:, t]
#                 logits = self.model(subseq, training=True)
#                 if t == self.num_sequences - 1:
#                     loss = self.loss_fn(y_batch, logits)
        
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#         return loss, logits

#     @tf.function
#     def test_statefull_step(self, x_batch, y_batch):
#         self.model.reset_states()
#         for t in range(self.num_sequences):
#             subseq = x_batch[:, t]
#             logits = self.model(subseq, training=False)
#             if t == self.num_sequences - 1:
#                 loss = self.loss_fn(y_batch, logits)
#         return loss, logits
    
#     def train_stateless(self, data: Tuple[tf.data.Dataset, tf.data.Dataset]) -> Dict:
#         """Entrenamiento stateless optimizado con estrategia seleccionada"""
#         train_data, test_data = data
#         batch_size = self.config.params['batch']
#         num_train_batches = len(list(train_data))  # Total de batches de entrenamiento
#         num_test_batches = len(list(test_data))    # Total de batches de test

#         train_metrics, test_metrics = TrainingUtils.init_metrics()
#         history = TrainingUtils.init_history()
#         start_time = time.time()
        
#         for epoch in range(self.config.params['epochs']):
#             epoch_start = time.time()
            
#             # --- Fase de Entrenamiento ---
#             train_metrics['loss'].reset_states()
#             train_metrics['acc'].reset_states()
            
#             print(f"\nEpoch {epoch+1}/{self.config.params['epochs']}")
#             train_progress = tqdm(train_data, desc="Training", unit="batch",total=num_train_batches, ncols=150)
            
#             for batch_idx, (batch, labels, _) in enumerate(train_progress):
#                 loss, logits = self.train_stateless_step(batch, labels)
                
#                 # Actualizar métricas
#                 train_metrics['batch_loss'] = loss.numpy()
#                 train_metrics['batch_acc'] = tf.reduce_mean(
#                     tf.keras.metrics.categorical_accuracy(labels, logits)).numpy()
#                 train_metrics['current_batch'] = batch_idx + 1
                
#                 train_metrics['loss'].update_state(loss)
#                 train_metrics['acc'].update_state(labels, logits)
                
#                 TrainingUtils.update_progress_bar(train_progress, train_metrics, 'train')
            
#             # --- Fase de Evaluación ---
#             test_metrics['loss'].reset_states()
#             test_metrics['acc'].reset_states()

#             # Agrupar secuencias por video
#             test_progress = tqdm(test_data, desc="Testing", unit="batch",
#                                 total=num_test_batches, ncols=100)
#             video_groups = {}
#             for batch_idx, (test_video_batch, test_label_batch, test_video_ids) in enumerate(test_progress):
#                 test_logits_batch = self.test_stateless_step(test_video_batch)
#                 for i, vid in enumerate(test_video_ids.numpy()):
#                     vid = vid.decode('utf-8') if isinstance(vid, bytes) else str(vid)
#                     if vid not in video_groups:
#                         video_groups[vid] = {'sequences': [], 'label': test_label_batch[i].numpy()}
#                     video_groups[vid]['sequences'].append(test_logits_batch[i].numpy())

#             video_test_progress = tqdm(video_groups.items(), desc="Evaluating videos")
#             # Evaluar cada video con la estrategia seleccionada
#             for vid, data in video_test_progress:
#                 sequences = np.array(data['sequences'])
#                 final_pred = VideoEvaluator.evaluate_video(sequences, self.strategy)

#                 # Actualizar métricas
#                 loss = self.loss_fn(tf.expand_dims(data['label'], 0),
#                                    tf.expand_dims(final_pred, 0))
#                 test_metrics['loss'].update_state(loss)
#                 test_metrics['acc'].update_state(data['label'], final_pred)

#                 TrainingUtils.update_progress_bar(video_test_progress, test_metrics, 'test')
            
#             # --- Registro de resultados ---
#             epoch_time = time.time() - epoch_start
#             TrainingUtils._update_history(history, train_metrics, test_metrics, epoch, epoch_time)
            
#             # --- Guardar mejor modelo ---
#             current_test_acc = test_metrics['acc'].result()
#             if current_test_acc > history['best_test_acc']:
#                 history['best_test_acc'] = current_test_acc
#                 self.model.save(f'{self.models_dir}/{self.model_tag}.h5')
#                 print(f"\nModel saved with test accuracy: {current_test_acc:.2%}")
        
#         return history['best_test_acc']
    
#     def train_statefull(self, data: Tuple[tf.data.Dataset, tf.data.Dataset]) -> Dict:
#         """Entrenamiento stateful optimizado con tf.function"""
#         train_data, test_data = data
#         batch_size = self.config.params['batch']
#         num_train_batches = len(list(train_data))
#         num_test_batches = len(list(test_data))

#         train_metrics, test_metrics = TrainingUtils.init_metrics()
#         history = TrainingUtils.init_history()
        
#         for epoch in range(self.config.params['epochs']):
#             epoch_start = time.time()
            
#             # --- Fase de Entrenamiento ---
#             train_metrics['loss'].reset_states()
#             train_metrics['acc'].reset_states()
            
#             print(f"\nEpoch {epoch+1}/{self.config.params['epochs']}")
#             train_progress = tqdm(train_data, desc="Training", unit="batch", total=num_train_batches, ncols=150)
            
#             for batch_idx, (video_batch, label_batch) in enumerate(train_progress):
#                 if video_batch.shape[0] < batch_size:
#                     continue
                    
#                 loss, logits = self.train_statefull_step( 
#                     video_batch, 
#                     label_batch, 
#                 )
                
#                 # Actualizar métricas (fuera del tf.function)
#                 train_metrics['batch_loss'] = loss.numpy()
#                 train_metrics['batch_acc'] = tf.reduce_mean(
#                     tf.keras.metrics.categorical_accuracy(label_batch, logits)).numpy()
#                 train_metrics['current_batch'] = batch_idx + 1
                
#                 train_metrics['loss'].update_state(loss)
#                 train_metrics['acc'].update_state(label_batch, logits)
                
#                 TrainingUtils.update_progress_bar(train_progress, train_metrics, 'train')
            
#             # --- Fase de Evaluación ---
#             test_metrics['loss'].reset_states()
#             test_metrics['acc'].reset_states()
            
#             test_progress = tqdm(test_data, desc="Testing", unit="batch", total=num_test_batches, ncols=150)
            
#             for batch_idx, (test_video_batch, test_label_batch) in enumerate(test_progress):
#                 if test_video_batch.shape[0] < batch_size:
#                     continue
                    
#                 t_loss, test_logits = self.test_statefull_step(
#                     test_video_batch,
#                     test_label_batch,
#                 )
                
#                 test_metrics['loss'].update_state(t_loss)
#                 test_metrics['acc'].update_state(test_label_batch, test_logits)
#                 test_metrics['current_batch'] = batch_idx + 1
                
#                 TrainingUtils.update_progress_bar(test_progress, test_metrics, 'test')
            
#             # --- Registro de resultados ---
#             epoch_time = time.time() - epoch_start
#             TrainingUtils._update_history(history, train_metrics, test_metrics, epoch, epoch_time)
            
#             # --- Guardar mejor modelo ---
#             current_test_acc = test_metrics['acc'].result()
#             if current_test_acc > history['best_test_acc']:
#                 history['best_test_acc'] = current_test_acc
#                 self.model.save(f'{self.models_dir}/{self.model_tag}.h5')
#                 print(f"\nModel saved with test accuracy: {current_test_acc:.2%}")
        
#         return history['best_test_acc']
    
#     def train(self, data):
#         if self.config.data['state'] == 'statefull':
#             return self.train_statefull(data)
#         elif self.config.data['state'] == 'stateless':
#             return self.train_stateless(data)
#         else:
#             raise ValueError("Estado del modelo no reconocido")


import os
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
from collections import Counter
from typing import Dict, List, Tuple
import sys  # Para forzar el flush de los prints

import gc

from config import print_memory


import objgraph
objgraph.show_growth()
class VideoEvaluator:
    """Clase optimizada para evaluación de videos con estrategia seleccionada"""

    @staticmethod
    @tf.function
    def evaluate_video(all_logits: np.ndarray,
                      strategy: str) -> np.ndarray:
        print(f"[VideoEvaluator] Evaluando video con estrategia: {strategy}", flush=True)
        if strategy == 'average':
            result = tf.reduce_mean(all_logits, axis=0)
        elif strategy == 'majority':
            result = VideoEvaluator._majority_vote(all_logits)
        elif strategy == 'max_prob':
            result = VideoEvaluator._max_probability(all_logits)
        elif strategy == 'temporal':
            result = VideoEvaluator._temporal_attention(all_logits)
        else:
            print(f"[VideoEvaluator] Estrategia no reconocida, usando promedio por defecto", flush=True)
            result = tf.reduce_mean(all_logits, axis=0)
        
        print(f"[VideoEvaluator] Evaluación completada", flush=True)
        return result
    @staticmethod
    @tf.function
    def _majority_vote(logits: tf.Tensor) -> tf.Tensor:
        predictions = tf.argmax(logits, axis=1)
        predictions = tf.cast(predictions, tf.int32)
        majority = tf.argmax(tf.math.bincount(predictions))
        return tf.one_hot(majority, logits.shape[1])

    @staticmethod
    @tf.function
    def _max_probability(logits: tf.Tensor) -> tf.Tensor:
        probs = tf.nn.softmax(logits)
        max_prob_index = tf.argmax(tf.reduce_max(probs, axis=1))
        return logits[max_prob_index]

    @staticmethod
    @tf.function
    def _temporal_attention(logits: tf.Tensor) -> tf.Tensor:
        weights = tf.linspace(0.1, 1.0, tf.shape(logits)[0])
        return tf.reduce_sum(logits * weights[:, tf.newaxis], axis=0) / tf.reduce_sum(weights)

class TrainingUtils:
    """Clase con utilidades comunes para ambos tipos de entrenamiento"""
    
    @staticmethod
    def init_metrics() -> Tuple[Dict, Dict]:
        print("[TrainingUtils] Inicializando métricas...", flush=True)
        train_metrics = {
            'loss': tf.keras.metrics.Mean(),
            'acc': tf.keras.metrics.CategoricalAccuracy(),
            'current_batch': 0,
            'batch_loss': 0.0,
            'batch_acc': 0.0
        }
        
        test_metrics = {
            'loss': tf.keras.metrics.Mean(),
            'acc': tf.keras.metrics.CategoricalAccuracy(),
            'current_batch': 0
        }
        
        return train_metrics, test_metrics
    
    @staticmethod
    def init_history() -> Dict:
        print("[TrainingUtils] Inicializando historial...", flush=True)
        return {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'time_per_epoch': [],
            'best_test_acc': 0.0
        }

    @staticmethod
    def _update_history(history: Dict, train_metrics: Dict, 
                       test_metrics: Dict, epoch: int, epoch_time: float):
        print(f"\n[TrainingUtils] Actualizando historial para época {epoch+1}", flush=True)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'].result().numpy())
        history['train_acc'].append(train_metrics['acc'].result().numpy())
        history['test_loss'].append(test_metrics['loss'].result().numpy())
        history['test_acc'].append(test_metrics['acc'].result().numpy())
        history['time_per_epoch'].append(epoch_time)
        
        print(f"[TrainingUtils] Época {epoch+1} - "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Train Acc: {history['train_acc'][-1]:.2%} - "
              f"Test Loss: {history['test_loss'][-1]:.4f}, "
              f"Test Acc: {history['test_acc'][-1]:.2%} - "
              f"Time: {epoch_time:.1f}s", flush=True)
    
    @staticmethod
    def update_progress_bar(progress_bar: tqdm, metrics: Dict, stage: str):
        if stage == 'train':
            progress_bar.set_postfix({
                'loss': f"{metrics['batch_loss']:.4f}",
                'acc': f"{metrics['batch_acc']:.2%}",
                'avg_loss': f"{metrics['loss'].result():.4f}",
                'avg_acc': f"{metrics['acc'].result():.2%}"
            })
        else:
            progress_bar.set_postfix({
                'loss': f"{metrics['loss'].result():.4f}",
                'acc': f"{metrics['acc'].result():.2%}"
            })

class Model():
    
    def __init__(self, config, num_sequences, num_features, num_classes):
        print(f"\n[Model] Inicializando modelo con configuración: {config}", flush=True)
        self.config = config
        self.num_sequences = num_sequences
        self.num_features = num_features
        self.num_classes = num_classes
        self.strategy = 'majority'
        self.sequences_tag = f"{self.config.data['cnn']}_{self.config.data['name']}_{self.config.data['frames']}_frames_{self.config.data['size']}_size_{self.config.data['seq']}_seq_{self.config.data['state']}"
        self.generate_model_tag()
        self.models_dir = f"{os.path.abspath(os.path.curdir)}/models/keras/{self.config.data['name']}/rnn/{self.config.data['cnn']}/{self.sequences_tag}/{self.model_tag}"
        print(f"[Model] Directorio de modelos: {self.models_dir}", flush=True)
        self.load_model()
    
    def generate_model_tag(self):
        print(f"[Model] Generando etiqueta del modelo...", flush=True)
        self.model_tag = f"{self.config.arch['type']}"
        for unit in self.config.arch['units']:
            self.model_tag += f"_{unit}u"
        self.model_tag += f"_{self.config.arch['direction']}"
        print(f"[Model] Etiqueta del modelo: {self.model_tag}", flush=True)

    def load_model(self):
        print(f"\n[Model] Cargando modelo para operación: {self.config.op}", flush=True)
        if self.config.op == 'train':
            self.load_train()
        elif self.config.op == 'eval':
            self.load_eval()
        else:
            print("[Model] ¡Error! Operación no reconocida", flush=True)
            sys.exit(1)

    def load_train(self):
        print("[Model] Configurando modelo para entrenamiento...", flush=True)
        recurrent_dropout=0
        dense_dropout=0.5
        dense_units=256  
        l2_reg=0.01

        units = tuple(u for u in self.config.arch['units'] if u not in {0})
        print(f"[Model] Unidades RNN: {units}", flush=True)
  
        batch_size = self.config.params['batch'] if self.config.data['state'] == 'statefull' else None
        print(f"[Model] Batch size: {batch_size}", flush=True)

        # Definición de la forma de entrada
        if self.config.data['state'] == 'stateless':
            input_shape = (self.config.data['seq'], self.num_features)
            inputs = tf.keras.Input(shape=input_shape, batch_size=None)
            print(f"[Model] Configuración stateless - input_shape: {input_shape}", flush=True)
        else:
            input_shape = (self.config.data['seq'], self.num_features)
            inputs = tf.keras.Input(batch_input_shape=(batch_size, *input_shape))
            print(f"[Model] Configuración stateful - batch_input_shape: {(batch_size, *input_shape)}", flush=True)

        RNNLayer = tf.keras.layers.LSTM if self.config.arch['type'] == 'lstm' else tf.keras.layers.GRU
        state = True if self.config.data['state'] == 'statefull' else False
        print(f"[Model] Tipo de capa RNN: {self.config.arch['type']}, stateful: {state}", flush=True)

        x = inputs

        for i, unit in enumerate(units):
            return_sequences = i < len(units) - 1
            print(f"[Model] Añadiendo capa RNN {i+1}/{len(units)} - unidades: {unit}, return_sequences: {return_sequences}", flush=True)
            
            recurrent_layer = RNNLayer(units=unit, return_sequences=return_sequences, stateful=state, recurrent_dropout=recurrent_dropout)
            
            if self.config.arch['direction'] == 'bidirectional':
                print("[Model] Añadiendo capa Bidirectional", flush=True)
                recurrent_layer = tf.keras.layers.Bidirectional(recurrent_layer)
            
            x = recurrent_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dropout(dense_dropout)(x)
        print("[Model] Añadiendo capa Dense intermedia", flush=True)
        x = tf.keras.layers.Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    
        print("[Model] Añadiendo capa Dense de salida", flush=True)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.config.params['lr'], clipvalue=1.0)
        print(f"[Model] Optimizador configurado: Adagrad con lr={self.config.params['lr']}", flush=True)
        
        print("[Model] Resumen del modelo:", flush=True)
        self.model.summary()
    
    def load_eval(self):
        print(f"[Model] Cargando modelo desde: {self.models_dir}/{self.model_tag}.h5", flush=True)
        try:
            self.model = tf.keras.models.load_model(f'{self.models_dir}/{self.model_tag}.h5')
            print("[Model] Modelo cargado exitosamente", flush=True)
        except Exception as e:
            print(f"[Model] ¡Error al cargar el modelo! {str(e)}", flush=True)
            sys.exit(1)

class ModelTrainer(Model):
    """Clase que contiene los métodos de entrenamiento optimizados"""
    
    def __init__(self, config, num_sequences, num_features, num_classes):
        print("\n[ModelTrainer] Inicializando entrenador...", flush=True)
        super().__init__(config, num_sequences, num_features, num_classes)
        print("[ModelTrainer] Entrenador inicializado", flush=True)
    
    @tf.function(experimental_relax_shapes=True)
    def train_stateless_step(self, batch, labels):
        """Paso de entrenamiento optimizado con tf.function"""
        with tf.GradientTape() as tape:
            logits = self.model(batch, training=True)
            loss = self.loss_fn(labels, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, logits
    
    @tf.function(experimental_relax_shapes=True)
    def test_stateless_step(self, batch):
        """Paso de evaluación optimizado con tf.function"""
        return self.model(batch, training=False)

    # @tf.function
    def train_statefull_step(self, x_batch, y_batch):
        self.model.reset_states()
        with tf.GradientTape() as tape:
            for t in range(self.num_sequences):
                subseq = x_batch[:, t]
                logits = self.model(subseq, training=True)
                if t == self.num_sequences - 1:
                    loss = self.loss_fn(y_batch, logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, logits

    # @tf.function
    def test_statefull_step(self, x_batch, y_batch):
        self.model.reset_states()
        for t in range(self.num_sequences):
            subseq = x_batch[:, t]
            logits = self.model(subseq, training=False)
            if t == self.num_sequences - 1:
                loss = self.loss_fn(y_batch, logits)
        return loss, logits
    
    def train_stateless(self, data: Tuple[tf.data.Dataset, tf.data.Dataset]) -> Dict:
        print("\n[ModelTrainer] Iniciando entrenamiento stateless", flush=True)
        train_data, test_data = data
        batch_size = self.config.params['batch']
        num_train_batches = len(list(train_data))
        num_test_batches = len(list(test_data))
        print(f"[ModelTrainer] Número de batches - Entrenamiento: {num_train_batches}, Test: {num_test_batches}", flush=True)

        train_metrics, test_metrics = TrainingUtils.init_metrics()
        history = TrainingUtils.init_history()
        start_time = time.time()
        
        for epoch in range(self.config.params['epochs']):
            print(f"\n[ModelTrainer] Iniciando época {epoch+1}/{self.config.params['epochs']}", flush=True)
            epoch_start = time.time()
            
            # --- Fase de Entrenamiento ---
            train_metrics['loss'].reset_states()
            train_metrics['acc'].reset_states()
            
            train_progress = tqdm(train_data, desc="Training", unit="batch", total=num_train_batches, ncols=150)
            
            for batch_idx, (batch, labels, _) in enumerate(train_progress):
                # print(f"[ModelTrainer] Procesando batch de entrenamiento {batch_idx+1}/{num_train_batches}", flush=True)
                try:
                    loss, logits = self.train_stateless_step(batch, labels)
                    
                    train_metrics['batch_loss'] = loss.numpy()
                    train_metrics['batch_acc'] = tf.reduce_mean(
                        tf.keras.metrics.categorical_accuracy(labels, logits)).numpy()
                    train_metrics['current_batch'] = batch_idx + 1
                    
                    train_metrics['loss'].update_state(loss)
                    train_metrics['acc'].update_state(labels, logits)
                    
                    TrainingUtils.update_progress_bar(train_progress, train_metrics, 'train')
                except Exception as e:
                    print(f"[ModelTrainer] ¡Error durante el entrenamiento! {str(e)}", flush=True)
                    raise
            
            # --- Fase de Evaluación ---
            print("[ModelTrainer] Iniciando fase de evaluación", flush=True)
            test_metrics['loss'].reset_states()
            test_metrics['acc'].reset_states()

            video_groups = {}
            test_progress = tqdm(test_data, desc="Testing", unit="batch", total=num_test_batches, ncols=100)
            
            for batch_idx, (test_video_batch, test_label_batch, test_video_ids) in enumerate(test_progress):
                # print(f"[ModelTrainer] Procesando batch de test {batch_idx+1}/{num_test_batches}", flush=True)
                try:
                    test_logits_batch = self.test_stateless_step(test_video_batch)
                    for i, vid in enumerate(test_video_ids.numpy()):
                        vid = vid.decode('utf-8') if isinstance(vid, bytes) else str(vid)
                        if vid not in video_groups:
                            video_groups[vid] = {'sequences': [], 'label': test_label_batch[i].numpy()}
                        video_groups[vid]['sequences'].append(test_logits_batch[i].numpy())
                except Exception as e:
                    print(f"[ModelTrainer] ¡Error durante la evaluación! {str(e)}", flush=True)
                    raise

            video_test_progress = tqdm(video_groups.items(), desc="Evaluating videos")
            for vid, data in video_test_progress:
                # print(f"[ModelTrainer] Evaluando video {vid}", flush=True)
                try:
                    sequences = np.array(data['sequences'])
                    final_pred = VideoEvaluator.evaluate_video(sequences, self.strategy)

                    loss = self.loss_fn(tf.expand_dims(data['label'], 0),
                                       tf.expand_dims(final_pred, 0))
                    test_metrics['loss'].update_state(loss)
                    test_metrics['acc'].update_state(data['label'], final_pred)

                    TrainingUtils.update_progress_bar(video_test_progress, test_metrics, 'test')
                except Exception as e:
                    print(f"[ModelTrainer] ¡Error evaluando video {vid}! {str(e)}", flush=True)
                    raise
            
            # --- Registro de resultados ---
            epoch_time = time.time() - epoch_start
            TrainingUtils._update_history(history, train_metrics, test_metrics, epoch, epoch_time)
            
            # --- Guardar mejor modelo ---
            current_test_acc = test_metrics['acc'].result()
            if current_test_acc > history['best_test_acc']:
                print(f"[ModelTrainer] ¡Nuevo mejor modelo encontrado! Accuracy: {current_test_acc:.2%}", flush=True)
                history['best_test_acc'] = current_test_acc
                try:
                    os.makedirs(self.models_dir, exist_ok=True)
                    self.model.save(f'{self.models_dir}/{self.model_tag}.h5')
                    print(f"[ModelTrainer] Modelo guardado en {self.models_dir}/{self.model_tag}.h5", flush=True)
                except Exception as e:
                    print(f"[ModelTrainer] ¡Error guardando el modelo! {str(e)}", flush=True)
                    raise
                    # --- Limpieza de memoria al final de la época ---
            print_memory()
            print(f"[ModelTrainer] Limpiando memoria después de la época {epoch+1}", flush=True)
            del batch, labels, test_video_batch, test_label_batch, test_video_ids
            del loss, logits, test_logits_batch, final_pred, sequences,
            tf.keras.backend.clear_session()
            gc.collect()
            print_memory()
        
        total_time = time.time() - start_time
        print(f"\n[ModelTrainer] Entrenamiento completado. Tiempo total: {total_time:.2f} segundos", flush=True)
        print(f"[ModelTrainer] Mejor precisión en test: {history['best_test_acc']:.2%}", flush=True)
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        del self.model, self.loss_fn, self.optimizer, test_data, train_data, data
        tf.keras.backend.clear_session()  # Limpia grafos persistentes
        gc.collect()   
           
        return history['best_test_acc']
    
    def train_statefull(self, data: Tuple[tf.data.Dataset, tf.data.Dataset]) -> Dict:
        objgraph.show_growth()
        print("\n[ModelTrainer] Iniciando entrenamiento stateful", flush=True)
        train_data, test_data = data
        batch_size = self.config.params['batch']
        num_train_batches = len(list(train_data))
        num_test_batches = len(list(test_data))
        print(f"[ModelTrainer] Número de batches - Entrenamiento: {num_train_batches}, Test: {num_test_batches}", flush=True)

        train_metrics, test_metrics = TrainingUtils.init_metrics()
        history = TrainingUtils.init_history()
        start_time = time.time()
        
        for epoch in range(self.config.params['epochs']):
            print(f"\n[ModelTrainer] Iniciando época {epoch+1}/{self.config.params['epochs']}", flush=True)
            epoch_start = time.time()
            # objgraph.show_growth()
            # --- Fase de Entrenamiento ---
            train_metrics['loss'].reset_states()
            train_metrics['acc'].reset_states()
            
            train_progress = tqdm(train_data, desc="Training", unit="batch", total=num_train_batches, ncols=150)
            # print_memory()
            for batch_idx, (video_batch, label_batch) in enumerate(train_progress):
                # print(f"[ModelTrainer] Procesando batch de entrenamiento {batch_idx+1}/{num_train_batches}", flush=True)
                if video_batch.shape[0] < batch_size:
                    print(f"[ModelTrainer] Batch demasiado pequeño ({video_batch.shape[0]} < {batch_size}), saltando...", flush=True)
                    continue
                    
                try:
                    # print("*"*50)
                    # print_memory()
                    loss, logits = self.train_statefull_step(video_batch, label_batch)
                    # print_memory()
                    train_metrics['batch_loss'] = loss.numpy()
                    train_metrics['batch_acc'] = tf.reduce_mean(
                        tf.keras.metrics.categorical_accuracy(label_batch, logits)).numpy()
                    train_metrics['current_batch'] = batch_idx + 1
                    
                    train_metrics['loss'].update_state(loss)
                    train_metrics['acc'].update_state(label_batch, logits)
                    # print_memory()
                    TrainingUtils.update_progress_bar(train_progress, train_metrics, 'train')
                    # print_memory()
                    # print("*"*50)
                except Exception as e:
                    print(f"[ModelTrainer] ¡Error durante el entrenamiento stateful! {str(e)}", flush=True)
                    raise
            
            # --- Fase de Evaluación ---
            print("[ModelTrainer] Iniciando fase de evaluación stateful", flush=True)
            test_metrics['loss'].reset_states()
            test_metrics['acc'].reset_states()
            
            test_progress = tqdm(test_data, desc="Testing", unit="batch", total=num_test_batches, ncols=150)
            # print_memory()
            for batch_idx, (test_video_batch, test_label_batch) in enumerate(test_progress):
                # print(f"[ModelTrainer] Procesando batch de test {batch_idx+1}/{num_test_batches}", flush=True)
                if test_video_batch.shape[0] < batch_size:
                    print(f"[ModelTrainer] Batch de test demasiado pequeño ({test_video_batch.shape[0]} < {batch_size}), saltando...", flush=True)
                    continue
                    
                try:
                    t_loss, test_logits = self.test_statefull_step(test_video_batch, test_label_batch)
                    # print_memory()
                    test_metrics['loss'].update_state(t_loss)
                    test_metrics['acc'].update_state(test_label_batch, test_logits)
                    test_metrics['current_batch'] = batch_idx + 1
                    # print_memory()
                    TrainingUtils.update_progress_bar(test_progress, test_metrics, 'test')
                    # print_memory()
                except Exception as e:
                    print(f"[ModelTrainer] ¡Error durante la evaluación stateful! {str(e)}", flush=True)
                    raise
            
            # --- Registro de resultados ---
            epoch_time = time.time() - epoch_start
            # print_memory()
            TrainingUtils._update_history(history, train_metrics, test_metrics, epoch, epoch_time)
            # print_memory()
            # --- Guardar mejor modelo ---
            current_test_acc = test_metrics['acc'].result()
            if current_test_acc > history['best_test_acc']:
                print(f"[ModelTrainer] ¡Nuevo mejor modelo encontrado! Accuracy: {current_test_acc:.2%}", flush=True)
                history['best_test_acc'] = current_test_acc
                try:
                    os.makedirs(self.models_dir, exist_ok=True)
                    self.model.save(f'{self.models_dir}/{self.model_tag}.h5')
                    print(f"[ModelTrainer] Modelo guardado en {self.models_dir}/{self.model_tag}.h5", flush=True)
                except Exception as e:
                    print(f"[ModelTrainer] ¡Error guardando el modelo! {str(e)}", flush=True)
                    raise
            ## print_memory()
            print(f"[ModelTrainer] Limpiando memoria después de la época {epoch+1}", flush=True)
            del video_batch, label_batch, test_video_batch, test_label_batch, loss, logits, t_loss, test_logits
            # # Forzar desasociación explícita del grafo
            # if hasattr(self, "train_statefull_step"):
            #     self.train_statefull_step = None

            # if hasattr(self, "test_statefull_step"):
            #     self.test_statefull_step = None
            # tf.keras.backend.clear_session()
            gc.collect()
            print_memory()

        total_time = time.time() - start_time
        print(f"\n[ModelTrainer] Entrenamiento stateful completado. Tiempo total: {total_time:.2f} segundos", flush=True)
        print(f"[ModelTrainer] Mejor precisión en test: {history['best_test_acc']:.2%}", flush=True)
        # Forzar desasociación explícita del grafo
        if hasattr(self, "train_statefull_step"):
            self.train_statefull_step = None

        if hasattr(self, "test_statefull_step"):
            self.test_statefull_step = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        del self.model, self.loss_fn, self.optimizer, test_data, train_data, data, test_progress, train_progress, test_metrics, train_metrics
        tf.keras.backend.clear_session()  # Limpia grafos persistentes   
        gc.collect()   
        
        return history['best_test_acc']
    
    def train(self, data):
        print("\n[ModelTrainer] Iniciando proceso de entrenamiento principal", flush=True)
        try:
            if self.config.data['state'] == 'statefull':
                print_memory
                print("[ModelTrainer] Modo stateful seleccionado", flush=True)
                result = self.train_statefull(data)
            elif self.config.data['state'] == 'stateless':
                print("[ModelTrainer] Modo stateless seleccionado", flush=True)
                result = self.train_stateless(data)
            else:
                raise ValueError("[ModelTrainer] Estado del modelo no reconocido")
            
            print("[ModelTrainer] Entrenamiento completado exitosamente", flush=True)
            return result
        except Exception as e:
            print(f"[ModelTrainer] ¡Error crítico durante el entrenamiento! {str(e)}", flush=True)
            raise