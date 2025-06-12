import os
import numpy as np
import tensorflow as tf

import time
from tqdm import tqdm  # Para barras de progreso (opcional)

from collections import Counter
from typing import Dict, List, Callable, Optional

class VideoClassificationStrategies:
    """Clase que contiene diferentes estrategias para clasificación de videos"""
    
    @staticmethod
    def average_logits(logits: List[np.ndarray]) -> np.ndarray:
        """Estrategia 1: Promedio de logits (default)"""
        return np.mean(logits, axis=0)
    
    @staticmethod
    def majority_vote(logits: List[np.ndarray]) -> np.ndarray:
        """Estrategia 2: Voto mayoritario de predicciones"""
        predictions = [np.argmax(logit) for logit in logits]
        majority = Counter(predictions).most_common(1)[0][0]
        one_hot = np.zeros_like(logits[0])
        one_hot[majority] = 1
        return one_hot
    
    @staticmethod
    def max_probability(logits: List[np.ndarray]) -> np.ndarray:
        """Estrategia 3: Selección de máxima probabilidad"""
        probs = [tf.nn.softmax(logit).numpy() for logit in logits]
        max_prob_index = np.argmax([np.max(prob) for prob in probs])
        return logits[max_prob_index]
    
    @staticmethod
    def temporal_attention(logits: List[np.ndarray]) -> np.ndarray:
        """Estrategia 4: Atención temporal ponderada"""
        weights = np.linspace(0.1, 1.0, num=len(logits))  # Más peso a frames finales
        return np.average(logits, axis=0, weights=weights)


class Model():
    
    def __init__(self, config, num_sequences, num_features, num_classes):
        self.config = config
        self.num_sequences = num_sequences
        self.num_features = num_features
        self.num_classes = num_classes
        self.strategy = 'majority'
        self.sequences_tag = f"{self.config.data['cnn']}_{self.config.data['name']}_{self.config.data['frames']}_frames_{self.config.data['size']}_size_{self.config.data['seq']}_seq_{self.config.data['state']}"
        self.generate_model_tag()
        self.models_dir = f"{os.path.abspath(os.path.curdir)}/models/keras/{self.config.data['name']}/rnn/{self.config.data['cnn']}/{self.sequences_tag}/{self.model_tag}"
        self.load_model()
        #tf.random.set_seed(264)

    def generate_model_tag(self):
        self.model_tag = f"{self.config.arch['type']}"
        for unit in self.config.arch['units']:
            self.model_tag += f"_{unit}u"
        self.model_tag += f"_{self.config.arch['direction']}"

    def load_train(self):
        recurrent_dropout=0.3
        dense_dropout=0.2
        dense_units=64 
        l2_reg=0.01

        units = tuple(u for u in self.config.arch['units'] if u not in {0})
  
        batch_size = self.config.params['batch'] if self.config.data['state'] == 'statefull' else None

        # Definición de la forma de entrada
        if self.config.data['state'] == 'stateless':
            input_shape = (self.config.data['seq'], self.num_features)
            inputs = tf.keras.Input(shape=input_shape, batch_size=None)  # Sin batch size fijo
        else:  # Stateful
            # Para stateful, debemos especificar el batch_size en el Input
            input_shape = (self.config.data['seq'], self.num_features)
            inputs = tf.keras.Input(batch_input_shape=(batch_size, *input_shape))  # batch_size incluido

        RNNLayer = tf.keras.layers.LSTM if self.config.arch['type'] == 'lstm' else tf.keras.layers.GRU
        state = True if self.config.data['state'] == 'statefull' else False

        x = inputs

        for i, unit in enumerate(units):
            return_sequences = i < len(units) - 1  # True excepto para la última capa
            
            recurrent_layer = RNNLayer(units=unit,return_sequences=return_sequences,stateful=state,recurrent_dropout=recurrent_dropout)
            # print(unit)
            if self.config.arch['direction'] == 'bidirectional':
                recurrent_layer = tf.keras.layers.Bidirectional(recurrent_layer)
            
            x = recurrent_layer(x)
            
            x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dropout(dense_dropout)(x)
        
        x = tf.keras.layers.Dense(dense_units,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)


        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.config.params['lr'],clipvalue=1.0)
        # self.modelcompile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=self.config.params['lr'],clipvalue=1.0),       #Adagrad(learning_rate=learning_rate),
        #                     loss=tf.keras.losses.CategoricalCrossentropy(),
        #                     metrics=['accuracy'])
        self.model.summary()
    
    def load_eval(self):
        self.model = tf.keras.models.load_model(f'{self.models_dir}/{self.model_tag}.h5')

    def load_model(self):
        if self.config.op == 'train':
            self.load_train()
        elif self.config.op == 'eval':
            self.load_eval()
        else:
            print("Error")

    def train(self, data):
        if self.config.data['state'] == 'statefull':
            test_acc = self.train_statefull(data)
        elif self.config.data['state'] == 'stateless':
            test_acc = self.train_stateless(data)
        else:
            print(f"error train: {self.config.data['state']}")
        return test_acc


    def _evaluate_video(self, sequences: np.ndarray) -> np.ndarray:
            """Evalúa un video completo usando la estrategia seleccionada"""
            # Obtener todas las predicciones
            dataset = tf.data.Dataset.from_tensor_slices(sequences)
            dataset = dataset.batch(self.config.params['batch'])
            
            all_logits = []
            for batch in dataset:
                logits = self.model(batch, training=False)
                all_logits.extend(logits.numpy())
            
            # Seleccionar estrategia
            strategies = {
                'average': VideoClassificationStrategies.average_logits,
                'majority': VideoClassificationStrategies.majority_vote,
                'max_prob': VideoClassificationStrategies.max_probability,
                'temporal': VideoClassificationStrategies.temporal_attention
            }
            
            strategy_fn = strategies.get(self.strategy, VideoClassificationStrategies.average_logits)
            return strategy_fn(all_logits)

    def train_stateless(self, data: tuple) -> Dict:
        """Entrenamiento para modelo stateless con múltiples estrategias de clasificación"""
        train_data, test_data = data
        batch_size = self.config.params['batch']
        num_train_batches = len(list(train_data))  # Total de batches de entrenamiento
        num_test_batches = len(list(test_data))    # Total de batches de test
        # Configuración
        metrics = {
            'train': {'loss': tf.keras.metrics.Mean(), 'acc': tf.keras.metrics.CategoricalAccuracy()},
            'test': {'loss': tf.keras.metrics.Mean(), 'acc': tf.keras.metrics.CategoricalAccuracy()}
        }
        
        history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [], 'time_per_epoch': []
        }
        
        best_test_acc = 0.0
        start_time = time.time()

        for epoch in range(self.config.params['epochs']):
            epoch_start = time.time()
            
            # --- Fase de Entrenamiento ---
            print(f"\nEpoch {epoch+1}/{self.config.params['epochs']}")
            train_progress = tqdm(train_data, desc="Training", unit="batch", 
                                total=num_train_batches, ncols=100)
            
            for batch, labels, _ in train_progress:
                with tf.GradientTape() as tape:
                    logits = self.model(batch, training=True)
                    loss = self.loss_fn(labels, logits)
                
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                metrics['train']['loss'].update_state(loss)
                metrics['train']['acc'].update_state(labels, logits)
                
                train_progress.set_postfix({
                    'loss': f"{metrics['train']['loss'].result():.4f}",
                    'acc': f"{metrics['train']['acc'].result():.2%}"
                })
            
            # --- Fase de Evaluación ---
            test_progress = tqdm(test_data, desc="Testing", unit="batch",
                                total=num_test_batches, ncols=100)
            video_groups = {}
            
            # Agrupar secuencias por video
            for batch, labels, video_ids in test_progress:
                for i, vid in enumerate(video_ids.numpy()):
                    vid = int(vid)
                    if vid not in video_groups:
                        video_groups[vid] = {'sequences': [], 'label': labels[i].numpy()}
                    video_groups[vid]['sequences'].append(batch[i].numpy())
            
            # Procesar cada video
            for vid, data in tqdm(video_groups.items(), desc="Evaluating videos"):
                sequences = np.array(data['sequences'])
                final_pred = self._evaluate_video(sequences)
                
                metrics['test']['acc'].update_state(data['label'], final_pred)
                loss = self.loss_fn(tf.expand_dims(data['label'], 0), 
                                  tf.expand_dims(final_pred, 0))
                metrics['test']['loss'].update_state(loss)
            
            # --- Registro de resultados ---
            epoch_time = time.time() - epoch_start
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(metrics['train']['loss'].result().numpy())
            history['train_acc'].append(metrics['train']['acc'].result().numpy())
            history['test_loss'].append(metrics['test']['loss'].result().numpy())
            history['test_acc'].append(metrics['test']['acc'].result().numpy())
            history['time_per_epoch'].append(epoch_time)
            
            print(f"\nEpoch {epoch+1} - "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Train Acc: {history['train_acc'][-1]:.2%} - "
                  f"Test Loss: {history['test_loss'][-1]:.4f}, "
                  f"Test Acc: {history['test_acc'][-1]:.2%} - "
                  f"Time: {epoch_time:.1f}s")
            
            # Guardar mejor modelo
            if history['test_acc'][-1] > best_test_acc:
                best_test_acc = history['test_acc'][-1]
                self.model.save(f"{self.models_dir}/{self.model_tag}.h5")
                print(f"Model saved with test accuracy: {best_test_acc:.2%}")
            
            # Reset métricas
            metrics['train']['loss'].reset_states()
            metrics['train']['acc'].reset_states()
            metrics['test']['loss'].reset_states()
            metrics['test']['acc'].reset_states()
        
        return best_test_acc

    # def train_stateless(self, data):
    #     train_data, test_data = data
        
    #     # Métricas
    #     metrics = {
    #         'train': {
    #             'loss': tf.keras.metrics.Mean(),
    #             'acc': tf.keras.metrics.CategoricalAccuracy(),
    #             'current_batch': 0,
    #             'batch_loss': 0.0,
    #             'batch_acc': 0.0
    #         },
    #         'test': {
    #             'loss': tf.keras.metrics.Mean(),
    #             'acc': tf.keras.metrics.CategoricalAccuracy(),
    #             'current_batch': 0
    #         }
    #     }

    #     # Historial completo
    #     history = {
    #         'epoch': [],
    #         'train_loss': [],
    #         'train_acc': [],
    #         'test_loss': [],
    #         'test_acc': [],
    #         'time_per_epoch': []
    #     }

    #     best_test_acc = 0.0
    #     start_time = time.time()

    #     # Clase para acumulación eficiente por video
    #     class VideoAccumulator:
    #         def __init__(self):
    #             self.preds = {}
    #             self.labels = {}
    #             self.counts = {}
            
    #         def update(self, video_ids, logits, labels):
    #             for i, vid in enumerate(video_ids.numpy()):
    #                 vid = vid.decode('utf-8') if isinstance(vid, bytes) else str(vid)
    #                 if vid not in self.preds:
    #                     self.preds[vid] = logits[i]
    #                     self.labels[vid] = labels[i]
    #                     self.counts[vid] = 1
    #                 else:
    #                     self.preds[vid] += logits[i]
    #                     self.counts[vid] += 1
            
    #         def get_metrics(self):
    #             avg_preds = tf.stack([self.preds[vid]/self.counts[vid] for vid in self.preds])
    #             true_labels = tf.stack([self.labels[vid] for vid in self.labels])
    #             return avg_preds, true_labels

    #     for epoch in range(self.config.params['epochs']):
    #         epoch_start_time = time.time()
    #         metrics['train']['loss'].reset_states()
    #         metrics['train']['acc'].reset_states()
            
    #         print(f"\n\033[1mEpoch {epoch+1}/{self.config.params['epochs']}\033[0m")
            
    #         # --- Entrenamiento ---
    #         train_accumulator = VideoAccumulator()
    #         train_progress = tqdm(train_data, desc="Training", unit="batch", ncols=100)

    #         @tf.function(experimental_relax_shapes=True)
    #         def train_step(batch, labels):
    #             with tf.GradientTape() as tape:
    #                 logits = self.model(batch, training=True)
    #                 loss = self.loss_fn(labels, logits)
    #             gradients = tape.gradient(loss, self.model.trainable_variables)
    #             self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    #             return loss, logits

    #         for batch_idx, (video_batch, label_batch, video_ids) in enumerate(train_progress):
    #             loss, logits = train_step(video_batch, label_batch)
                
    #             # Acumulación optimizada
    #             train_accumulator.update(video_ids, logits, label_batch)
                
    #             # Actualizar métricas del batch
    #             metrics['train']['batch_loss'] = loss.numpy()
    #             batch_acc = tf.reduce_mean(
    #                 tf.keras.metrics.categorical_accuracy(label_batch, logits))
    #             metrics['train']['batch_acc'] = batch_acc.numpy()
    #             metrics['train']['current_batch'] = batch_idx + 1
                
    #             train_progress.set_postfix({
    #                 'batch_loss': f"{loss.numpy():.4f}",
    #                 'batch_acc': f"{batch_acc:.2%}"
    #             })

    #         # Calcular métricas finales de entrenamiento
    #         avg_preds, true_labels = train_accumulator.get_metrics()
    #         metrics['train']['loss'].update_state(self.loss_fn(true_labels, avg_preds))
    #         metrics['train']['acc'].update_state(true_labels, avg_preds)

    #         # --- Evaluación ---
    #         metrics['test']['loss'].reset_states()
    #         metrics['test']['acc'].reset_states()
    #         test_accumulator = VideoAccumulator()
    #         test_progress = tqdm(test_data, desc="Testing", unit="batch", ncols=100)

    #         @tf.function(experimental_relax_shapes=True)
    #         def test_step(batch):
    #             return self.model(batch, training=False)

    #         for batch_idx, (test_video_batch, test_label_batch, test_video_ids) in enumerate(test_progress):
    #             test_logits = test_step(test_video_batch)
    #             test_accumulator.update(test_video_ids, test_logits, test_label_batch)
    #             test_progress.set_postfix({'progress': f"{batch_idx+1}/{len(list(test_data))}"})

    #         # Calcular métricas finales de test
    #         test_avg_preds, test_true_labels = test_accumulator.get_metrics()
    #         metrics['test']['loss'].update_state(self.loss_fn(test_true_labels, test_avg_preds))
    #         metrics['test']['acc'].update_state(test_true_labels, test_avg_preds)

    #         # --- Cálculo de tiempos ---
    #         epoch_time = time.time() - epoch_start_time
    #         total_time = time.time() - start_time
    #         time_remaining = (self.config.params['epochs'] - epoch - 1) * epoch_time
            
    #         # --- Logging ---
    #         print("\n\033[1mResumen Época:\033[0m")
    #         print(f"• Entrenamiento: Loss = {metrics['train']['loss'].result():.4f} | Acc = {metrics['train']['acc'].result():.2%}")
    #         print(f"• Validación:    Loss = {metrics['test']['loss'].result():.4f} | Acc = {metrics['test']['acc'].result():.2%}")
    #         print(f"• Tiempo: {epoch_time:.1f}s (Total: {total_time:.1f}s | Restante: ~{time_remaining:.1f}s)")
            
    #         # Guardar en historial
    #         history['epoch'].append(epoch + 1)
    #         history['train_loss'].append(metrics['train']['loss'].result().numpy())
    #         history['train_acc'].append(metrics['train']['acc'].result().numpy())
    #         history['test_loss'].append(metrics['test']['loss'].result().numpy())
    #         history['test_acc'].append(metrics['test']['acc'].result().numpy())
    #         history['time_per_epoch'].append(epoch_time)
            
    #         # --- Guardar mejor modelo ---
    #         current_test_acc = metrics['test']['acc'].result()
    #         if current_test_acc > best_test_acc:
    #             best_test_acc = current_test_acc
    #             self.model.save(f'{self.models_dir}/{self.model_tag}.h5')
    #             print(f"\n\033[32mModelo guardado (Mejor accuracy: {best_test_acc:.2%})\033[0m")

    #     return history



    def train_statefull(self, data):
        train_data, test_data = data
        batch_size = self.config.params['batch']
        num_train_batches = len(list(train_data))  # Total de batches de entrenamiento
        num_test_batches = len(list(test_data))    # Total de batches de test
        
        # Métricas
        metrics = {
            'train': {
                'loss': tf.keras.metrics.Mean(),
                'acc': tf.keras.metrics.CategoricalAccuracy(),
                'current_batch': 0,
                'batch_loss': 0.0,
                'batch_acc': 0.0
            },
            'test': {
                'loss': tf.keras.metrics.Mean(),
                'acc': tf.keras.metrics.CategoricalAccuracy(),
                'current_batch': 0
            }
        }

        # Historial completo
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'time_per_epoch': []
        }

        best_test_acc = 0.0
        start_time = time.time()

        for epoch in range(self.config.params['epochs']):
            epoch_start_time = time.time()
            metrics['train']['loss'].reset_states()
            metrics['train']['acc'].reset_states()
            
            print(f"\n\033[1mEpoch {epoch+1}/{self.config.params['epochs']}\033[0m")
            
            # --- Entrenamiento con barra de progreso ---
            train_progress = tqdm(train_data, desc="Training", unit="batch", 
                                total=num_train_batches, ncols=100)
            
            for batch_idx, (video_batch, label_batch) in enumerate(train_progress):
                if video_batch.shape[0] < batch_size:
                    continue
                    
                self.model.reset_states()
                
                # Procesar secuencia temporal
                for t in range(self.num_sequences):
                    subseq = video_batch[:, t]
                    is_last_step = (t == self.num_sequences - 1)
                    
                    with tf.GradientTape() as tape:
                        logits = self.model(subseq, training=True)
                        if is_last_step:
                            loss = self.loss_fn(label_batch, logits)
                    
                    if is_last_step:
                        gradients = tape.gradient(loss, self.model.trainable_variables)
                        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                        
                        # Actualizar métricas del batch actual
                        metrics['train']['batch_loss'] = loss.numpy()
                        metrics['train']['batch_acc'] = tf.keras.metrics.categorical_accuracy(
                            label_batch, logits).numpy().mean()
                        metrics['train']['current_batch'] = batch_idx + 1
                        
                        # Actualizar métricas acumuladas
                        metrics['train']['loss'].update_state(loss)
                        metrics['train']['acc'].update_state(label_batch, logits)
                
                # Actualizar barra de progreso con info en tiempo real
                train_progress.set_postfix({
                    'loss': f"{metrics['train']['batch_loss']:.4f}",
                    'acc': f"{metrics['train']['batch_acc']:.2%}",
                    'avg_loss': f"{metrics['train']['loss'].result():.4f}",
                    'avg_acc': f"{metrics['train']['acc'].result():.2%}"
                })

            # --- Evaluación ---
            metrics['test']['loss'].reset_states()
            metrics['test']['acc'].reset_states()
            self.model.reset_states()
            
            test_progress = tqdm(test_data, desc="Testing", unit="batch",
                                total=num_test_batches, ncols=100)
            
            for batch_idx, (test_video_batch, test_label_batch) in enumerate(test_progress):
                if test_video_batch.shape[0] < batch_size:
                    continue
                    
                self.model.reset_states()
                
                for t in range(self.num_sequences):
                    test_subseq = test_video_batch[:, t]
                    test_logits = self.model(test_subseq, training=False)
                    
                    if t == self.num_sequences - 1:
                        t_loss = self.loss_fn(test_label_batch, test_logits)
                        metrics['test']['loss'].update_state(t_loss)
                        metrics['test']['acc'].update_state(test_label_batch, test_logits)
                        metrics['test']['current_batch'] = batch_idx + 1
                
                test_progress.set_postfix({
                    'loss': f"{metrics['test']['loss'].result():.4f}",
                    'acc': f"{metrics['test']['acc'].result():.2%}"
                })

            # --- Cálculo de tiempos ---
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            time_remaining = (self.config.params['epochs'] - epoch - 1) * epoch_time
            
            # --- Logging detallado ---
            print("\n\033[1mResumen Época:\033[0m")
            print(f"• Entrenamiento: Loss = {metrics['train']['loss'].result():.4f} | Acc = {metrics['train']['acc'].result():.2%}")
            print(f"• Validación:    Loss = {metrics['test']['loss'].result():.4f} | Acc = {metrics['test']['acc'].result():.2%}")
            print(f"• Tiempo: {epoch_time:.1f}s (Total: {total_time:.1f}s | Restante: ~{time_remaining:.1f}s)")
            
            # Guardar en historial
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(metrics['train']['loss'].result().numpy())
            history['train_acc'].append(metrics['train']['acc'].result().numpy())
            history['test_loss'].append(metrics['test']['loss'].result().numpy())
            history['test_acc'].append(metrics['test']['acc'].result().numpy())
            history['time_per_epoch'].append(epoch_time)
            
            # --- Guardar mejor modelo ---
            current_test_acc = metrics['test']['acc'].result()
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                self.model.save(f'{self.models_dir}/{self.model_tag}.h5')
                print(f"\n\033[32mModelo guardado (Mejor accuracy: {best_test_acc:.2%})\033[0m")

        return best_test_acc

    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de prueba, manejando tanto stateless como stateful.
        
        Args:
            test_data: Dataset de prueba que contiene (batch, labels, video_ids) para stateless
                    o (batch, labels) para stateful
        
        Returns:
            Dict con las métricas de evaluación (loss y accuracy)
        """
        metrics = {
            'loss': tf.keras.metrics.Mean(),
            'acc': tf.keras.metrics.CategoricalAccuracy()
        }
        
        if self.config.data['state'] == 'statefull':
            return self._evaluate_statefull(test_data, metrics)
        elif self.config.data['state'] == 'stateless':
            return self._evaluate_stateless(test_data, metrics)
        else:
            raise ValueError("Estado del modelo no reconocido")

    def _evaluate_stateless(self, test_data: tf.data.Dataset, metrics: Dict) -> Dict[str, float]:
        """Evaluación para modelos stateless con estrategias de clasificación de video"""
        video_groups = {}
        
        # Agrupar secuencias por video
        for batch, labels, video_ids in test_data:
            for i, vid in enumerate(video_ids.numpy()):
                vid = int(vid)
                if vid not in video_groups:
                    video_groups[vid] = {'sequences': [], 'label': labels[i].numpy()}
                video_groups[vid]['sequences'].append(batch[i].numpy())
        
        # Procesar cada video
        for vid, data in tqdm(video_groups.items(), desc="Evaluating videos"):
            sequences = np.array(data['sequences'])
            final_pred = self._evaluate_video(sequences)
            
            metrics['acc'].update_state(data['label'], final_pred)
            loss = self.loss_fn(tf.expand_dims(data['label'], 0), 
                            tf.expand_dims(final_pred, 0))
            metrics['loss'].update_state(loss)
        
        return {
            'loss': metrics['loss'].result().numpy(),
            'accuracy': metrics['acc'].result().numpy()
        }

    def _evaluate_statefull(self, test_data: tf.data.Dataset, metrics: Dict) -> Dict[str, float]:
        """Evaluación para modelos stateful"""
        batch_size = self.config.params['batch']
        num_test_batches = len(list(test_data))
        
        test_progress = tqdm(test_data, desc="Testing", unit="batch",
                            total=num_test_batches, ncols=100)
        
        for batch_idx, (test_video_batch, test_label_batch) in enumerate(test_progress):
            if test_video_batch.shape[0] < batch_size:
                continue
                
            self.model.reset_states()
            
            for t in range(self.num_sequences):
                test_subseq = test_video_batch[:, t]
                test_logits = self.model(test_subseq, training=False)
                
                if t == self.num_sequences - 1:
                    t_loss = self.loss_fn(test_label_batch, test_logits)
                    metrics['loss'].update_state(t_loss)
                    metrics['acc'].update_state(test_label_batch, test_logits)
            
            test_progress.set_postfix({
                'loss': f"{metrics['loss'].result():.4f}",
                'acc': f"{metrics['acc'].result():.2%}"
            })
        
        return {
            'loss': metrics['loss'].result().numpy(),
            'accuracy': metrics['acc'].result().numpy()
        }