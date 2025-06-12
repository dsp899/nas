
import os
import numpy as np
import tensorflow
import time
import datetime

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback

from LSTMSearchSpace import LSTMSearchSpace

from CONSTANTS import *

  
 
class Controller(LSTMSearchSpace):

    def __init__(self):
        self.max_len = SEQUENCE_LENGTH
        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM

        self.controller_weights = f"{os.path.abspath(os.path.curdir)}/results/{DATASET_NAME}/nas_{CONTROLLER_SAMPLES_PER_EPOCH}_samples_{CONTROLLER_SAMPLING_EPOCHS}_epochs_{FRAMES}_frames.h5"

        self.seq_data = []

        super().__init__()

        self.controller_classes = len(self.vocab)

    # def sample_architecture_sequences(self, model, number_of_samples):
    #     samples = []
        
    #     # Obtener vocabularios
    #     (vocab, vocab_layers, vocab_seq, vocab_cnn, vocab_rnn, 
    #     vocab_units_0, vocab_units_1, vocab_units_2, 
    #     vocab_direction, vocab_state) = self.vocab_dict()
        
    #     # Extraer IDs
    #     layers_ids = list(vocab_layers.keys())
    #     seq_ids = list(vocab_seq.keys())
    #     cnn_ids = list(vocab_cnn.keys())
    #     rnn_ids = list(vocab_rnn.keys())
    #     units_0_ids = list(vocab_units_0.keys())
    #     units_1_ids = list(vocab_units_1.keys())
    #     units_2_ids = list(vocab_units_2.keys())
    #     direction_ids = list(vocab_direction.keys())
    #     state_ids = list(vocab_state.keys())
        
    #     # Índices especiales
    #     units_1_zero_idx = units_1_ids[0]
    #     units_2_zero_idx = units_2_ids[0]
        
    #     def apply_layer_mask(num_layers, probab):
    #         masked_probab = np.copy(probab)
            
    #         if num_layers == 1:
    #             for idx in units_1_ids[1:]:
    #                 masked_probab[idx] = 0
    #             for idx in units_2_ids[1:]:
    #                 masked_probab[idx] = 0
    #         elif num_layers == 2:
    #             masked_probab[units_1_zero_idx] = 0
    #             for idx in units_2_ids[1:]:
    #                 masked_probab[idx] = 0
    #         elif num_layers == 3:
    #             masked_probab[units_1_zero_idx] = 0
    #             masked_probab[units_2_zero_idx] = 0
            
    #         # Renormalización robusta
    #         prob_sum = np.sum(masked_probab)
    #         if prob_sum > 0:
    #             masked_probab = masked_probab / prob_sum
    #         else:
    #             masked_probab = np.ones_like(masked_probab) / len(masked_probab)
            
    #         # Asegurar suma exactamente 1.0
    #         masked_probab[-1] += 1.0 - np.sum(masked_probab)
    #         return masked_probab
        
    #     def safe_random_choice(valid_ids, probs):
    #         """Muestreo seguro que maneja errores de precisión numérica"""
    #         probs = np.asarray(probs)
    #         probs = probs / probs.sum()  # Renormalizar por si acaso
    #         return np.random.choice(valid_ids, p=probs)
        
    #     while len(samples) < number_of_samples:
    #         sample = []
    #         sequence = pad_sequences([sample], maxlen=self.max_len-1, padding='post')
    #         sequence = sequence.reshape(1, self.max_len - 1, 1)
    #         probab = model.predict(sequence, verbose=0)
    #         print(f"Model predict probab shape: {probab.shape}##############################################")
    #         probab = probab[0][-1]
    #         while len(sample) < self.max_len:

                
    #             # Paso 1: Predecir num_layers
    #             if len(sample) == 0:
    #                 layer_probs = probab[layers_ids]
    #                 layer_probs = layer_probs / layer_probs.sum()  # Renormalizar
    #                 num_layers = safe_random_choice(layers_ids, layer_probs)
    #                 sample.append(num_layers)  # Guardar como índice (0, 1, 2)
    #                 continue
                
    #             # Paso 2: Aplicar máscaras
    #             current_num_layers = sample[0] + 1
    #             masked_probab = apply_layer_mask(current_num_layers, probab)
                
    #             # Paso 3: Muestrear tokens condicionalmente
    #             try:
    #                 if len(sample) == 1:  # seq
    #                     probs = masked_probab[seq_ids]
    #                     next_token = safe_random_choice(seq_ids, probs)
    #                     sample.append(next_token)
                    
    #                 elif len(sample) == 2:  # cnn
    #                     probs = masked_probab[cnn_ids]
    #                     next_token = safe_random_choice(cnn_ids, probs)
    #                     sample.append(next_token)
                    
    #                 elif len(sample) == 3:  # rnn
    #                     probs = masked_probab[rnn_ids]
    #                     next_token = safe_random_choice(rnn_ids, probs)
    #                     sample.append(next_token)
                    
    #                 elif len(sample) == 4:  # units_0
    #                     probs = masked_probab[units_0_ids]
    #                     next_token = safe_random_choice(units_0_ids, probs)
    #                     sample.append(next_token)
                    
    #                 elif len(sample) == 5:  # units_1
    #                     probs = masked_probab[units_1_ids]
    #                     next_token = safe_random_choice(units_1_ids, probs)
    #                     sample.append(next_token)
                    
    #                 elif len(sample) == 6:  # units_2
    #                     probs = masked_probab[units_2_ids]
    #                     next_token = safe_random_choice(units_2_ids, probs)
    #                     sample.append(next_token)
                    
    #                 elif len(sample) == 7:  # direction
    #                     probs = masked_probab[direction_ids]
    #                     next_token = safe_random_choice(direction_ids, probs)
    #                     sample.append(next_token)
                    
    #                 elif len(sample) == 8:  # state
    #                     probs = masked_probab[state_ids]
    #                     next_token = safe_random_choice(state_ids, probs)
    #                     sample.append(next_token)
                
    #             except ValueError as e:
    #                 print(f"Error en muestreo: {e}")
    #                 print(f"Probs: {probs}")
    #                 print(f"Suma: {np.sum(probs)}")
    #                 raise
                
    #         if sample not in self.seq_data:
    #             samples.append(sample)
    #             self.seq_data.append(sample)
        
    #     return samples



    def sample_architecture_sequences(self, model, number_of_samples):
        samples = []

        # Obtener vocabularios
        (vocab, vocab_layers, vocab_seq, vocab_cnn, vocab_rnn, 
            vocab_units_0, vocab_units_1, vocab_units_2, 
            vocab_direction, vocab_state) = self.vocab_dict()

        # Extraer IDs
        layers_ids = list(vocab_layers.keys())
        seq_ids = list(vocab_seq.keys())
        cnn_ids = list(vocab_cnn.keys())
        rnn_ids = list(vocab_rnn.keys())
        units_0_ids = list(vocab_units_0.keys())
        units_1_ids = list(vocab_units_1.keys())
        units_2_ids = list(vocab_units_2.keys())
        direction_ids = list(vocab_direction.keys())
        state_ids = list(vocab_state.keys())

        # Índices especiales
        units_1_zero_idx = units_1_ids[0]
        units_2_zero_idx = units_2_ids[0]

        # def apply_layer_mask(num_layers, probab):
        #     """Aplica máscara a las probabilidades según el número de capas."""
        #     masked_probab = np.copy(probab)
        #     masked_probab = np.maximum(masked_probab, 0)  # Forzar no-negatividad
            
        #     if num_layers == 1:
        #         masked_probab[units_1_ids[1:]] = 0
        #         masked_probab[units_2_ids[1:]] = 0
        #     elif num_layers == 2:
        #         masked_probab[units_1_zero_idx] = 0
        #         masked_probab[units_2_ids[1:]] = 0
        #     elif num_layers == 3:
        #         masked_probab[units_1_zero_idx] = 0
        #         masked_probab[units_2_zero_idx] = 0
            
        #     # Renormalización robusta
        #     prob_sum = np.sum(masked_probab)
        #     if prob_sum > 1e-10:  # Evitar división por cero
        #         masked_probab = masked_probab / prob_sum
        #     else:
        #         masked_probab = np.ones_like(masked_probab) / len(masked_probab)
            
        #     return masked_probab

        # def safe_random_choice(valid_ids, probs):
        #     """Muestreo seguro con manejo de errores numéricos."""
        #     probs = np.asarray(probs)
        #     probs = np.maximum(probs, 0)  # Eliminar negativos
        #     probs = probs / (probs.sum() + 1e-10)  # Renormalizar con estabilidad
            
        #     # Validación final
        #     if np.any(probs < 0) or abs(probs.sum() - 1.0) > 0.1:#1e-5:
        #         print(f"¡Advertencia: Probabilidades inválidas! Suma: {probs.sum()}")
        #         probs = np.ones_like(probs) / len(probs)  # Distribución uniforme como fallback
            
        #     return np.random.choice(valid_ids, p=probs)



        def apply_layer_mask(num_layers, probab):
            """Aplica máscara a las probabilidades según el número de capas."""
            masked_probab = np.copy(probab)
            masked_probab = np.maximum(masked_probab, 0)
            valid_mask = np.ones_like(masked_probab, dtype=bool)

            if num_layers == 1:
                masked_probab[units_1_ids[1:]] = 0
                masked_probab[units_2_ids[1:]] = 0
                valid_mask[units_1_ids[1:]] = False
                valid_mask[units_2_ids[1:]] = False
            elif num_layers == 2:
                masked_probab[units_1_zero_idx] = 0
                masked_probab[units_2_ids[1:]] = 0
                valid_mask[units_1_zero_idx] = False
                valid_mask[units_2_ids[1:]] = False
            elif num_layers == 3:
                masked_probab[units_1_zero_idx] = 0
                masked_probab[units_2_zero_idx] = 0
                valid_mask[units_1_zero_idx] = False
                valid_mask[units_2_zero_idx] = False

            prob_sum = masked_probab.sum()
            if prob_sum > 1e-10:
                masked_probab /= prob_sum
            else:
                masked_probab[:] = 0
                masked_probab[valid_mask] = 1.0 / valid_mask.sum()

            return masked_probab


        def safe_random_choice(valid_ids, probs):
            """Muestreo seguro que evita fallos duros y aplica fallback si es necesario."""
            probs = np.asarray(probs, dtype=np.float64)
            probs = np.maximum(probs, 0)
            total = probs.sum()

            if total < 1e-10 or len(valid_ids) == 0:
                print("⚠️ [safe_random_choice] Probabilidades inválidas. Fallback aleatorio uniforme.")
                return np.random.choice(valid_ids)

            try:
                probs = probs / total
                probs[-1] += 1.0 - probs.sum()  # corrección mínima
                return np.random.choice(valid_ids, p=probs)
            except Exception as e:
                print(f"⚠️ [safe_random_choice] Error en muestreo con p. Fallback uniforme. Error: {e}")
                return np.random.choice(valid_ids)


        while len(samples) < number_of_samples:
            sample = []
            sequence = pad_sequences([sample], maxlen=self.max_len-1, padding='post')
            sequence = sequence.reshape(1, self.max_len - 1, 1)
            
            # Paso 0: Predecir y aplicar softmax
            probab  = model.predict(sequence, verbose=0)[0][-1]  # Asume que el modelo devuelve logits
            # probab = np.exp(logits) / np.sum(np.exp(logits))  # Softmax manual
            
            while len(sample) < self.max_len:
                try:
                    # Paso 1: Predecir num_layers (primer token)
                    if len(sample) == 0:
                        layer_probs = probab[layers_ids]
                        num_layers = safe_random_choice(layers_ids, layer_probs)
                        sample.append(num_layers)
                        continue
                    
                    # Paso 2: Aplicar máscara según num_layers
                    current_num_layers = sample[0] + 1  # Asume que sample[0] es el índice de capas
                    masked_probab = apply_layer_mask(current_num_layers, probab)
    

                    # Paso 3: Muestrear tokens condicionalmente (con seq y cnn al final)
                    if len(sample) == 1:  # rnn
                        next_token = safe_random_choice(rnn_ids, masked_probab[rnn_ids])
                    elif len(sample) == 2:  # units_0
                        next_token = safe_random_choice(units_0_ids, masked_probab[units_0_ids])
                    elif len(sample) == 3:  # units_1
                        next_token = safe_random_choice(units_1_ids, masked_probab[units_1_ids])
                    elif len(sample) == 4:  # units_2
                        next_token = safe_random_choice(units_2_ids, masked_probab[units_2_ids])
                    elif len(sample) == 5:  # direction
                        next_token = safe_random_choice(direction_ids, masked_probab[direction_ids])
                    elif len(sample) == 6:  # state
                        next_token = safe_random_choice(state_ids, masked_probab[state_ids])
                    elif len(sample) == 7:  # seq (ahora es el penúltimo)
                        next_token = safe_random_choice(seq_ids, masked_probab[seq_ids])
                    elif len(sample) == 8:  # cnn (ahora es el último)
                        next_token = safe_random_choice(cnn_ids, masked_probab[cnn_ids])
                    
                    sample.append(next_token)
            # while len(sample) < self.max_len:
            #     try:
            #         # Paso 1: Predecir seq (primer token)
            #         if len(sample) == 0:
            #             seq_probs = probab[seq_ids]
            #             next_token = safe_random_choice(seq_ids, seq_probs)
            #             sample.append(next_token)
            #             continue

            #         # Paso 2: Predecir cnn (segundo token)
            #         if len(sample) == 1:
            #             cnn_probs = probab[cnn_ids]
            #             next_token = safe_random_choice(cnn_ids, cnn_probs)
            #             sample.append(next_token)
            #             continue

            #         # Paso 3: Predecir layers (tercer token)
            #         if len(sample) == 2:
            #             layer_probs = probab[layers_ids]
            #             num_layers = safe_random_choice(layers_ids, layer_probs)
            #             sample.append(num_layers)
            #             continue

            #         # Aplicar máscara según número de capas
            #         current_num_layers = sample[2] + 1  # ahora sample[2] es el índice de capas
            #         masked_probab = apply_layer_mask(current_num_layers, probab)

            #         # Paso 4 en adelante: muestreo condicional del resto
            #         if len(sample) == 3:  # rnn
            #             next_token = safe_random_choice(rnn_ids, masked_probab[rnn_ids])
            #         elif len(sample) == 4:  # units_0
            #             next_token = safe_random_choice(units_0_ids, masked_probab[units_0_ids])
            #         elif len(sample) == 5:  # units_1
            #             next_token = safe_random_choice(units_1_ids, masked_probab[units_1_ids])
            #         elif len(sample) == 6:  # units_2
            #             next_token = safe_random_choice(units_2_ids, masked_probab[units_2_ids])
            #         elif len(sample) == 7:  # direction
            #             next_token = safe_random_choice(direction_ids, masked_probab[direction_ids])
            #         elif len(sample) == 8:  # state
            #             next_token = safe_random_choice(state_ids, masked_probab[state_ids])

            #         sample.append(next_token)

                
                except Exception as e:
                    print(f"Error en muestreo: {e}")
                    print(f"Probabilidades problemáticas: {masked_probab if 'masked_probab' in locals() else probab}")
                    print(f"Suma: {np.sum(masked_probab if 'masked_probab' in locals() else probab)}")
                    raise
            
            # Evitar duplicados
            if sample not in self.seq_data:
                samples.append(sample)
                self.seq_data.append(sample)

        return samples
        

    def create_controller_model(self, controller_input_shape, controller_batch_size):

        main_input = Input(shape=controller_input_shape, name='main_input')

        x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)

        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)

        controller_model = Model(inputs=[main_input], outputs=[main_output])
		
        controller_model.summary()
        
        return controller_model

   
    def train_controller(self, model, x_data, y_data, loss_func, nb_epochs):

        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.controller_lr, momentum=self.controller_momentum, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(learning_rate=self.controller_lr, clipnorm=1.0)

        model.compile(optimizer=optim, loss={'main_output': loss_func})

        # if os.path.exists(self.controller_weights):
        #     model.load_weights(self.controller_weights)

        # model.summary()
        print(f"Fit Controller################################################################################################################ ")
        history = model.fit({'main_input': x_data},
                  {'main_output': y_data},
                  epochs=nb_epochs)
                  # verbose=0, callbacks=[MyCustomCallback(x_data,y_data.reshape(len(y_data), 1, self.controller_classes)) ])   #verbose=0
        
        print(f"history keys:{history.history.keys()}:{history.history['loss']}")

        # model.summary()
        
        # for layer in model.layers:
        #   print("layer: ", layer.name)        
        #   weights = layer.get_weights()
        #   print("weights: ", weights)
        # model.save_weights(self.controller_weights)
        
        # model.load_weights(self.controller_weights)
        return history
        


