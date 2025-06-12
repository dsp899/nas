from controller import Controller
from CONSTANTS import *
from utils import *
#from manage_cnn import *
# from manage_lstm import *
from run_rnn import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import config
# import load_data
# import load_model_rnn
import keras.backend as K
import tensorflow as tf

import json
from datetime import datetime

from config import print_memory

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para manejar tipos numpy en JSON"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MYLSTMNAS(Controller):

    def __init__(self):
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.controller_samples_per_epoch = CONTROLLER_SAMPLES_PER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA

        self.data = []
        self.controller_rewards = []  # Lista para almacenar rewards por época
        self.controller_losses = []   # Lista para almacenar losses por época
        self.discounted_rewards = []  # Lista para almacenar discounted rewards
        


        self.nas_data_log = 'LOGS/nas_data.pkl'
        
        clean_log()

        super().__init__()

        self.analysis_file = f"{os.path.abspath(os.path.curdir)}/results/{DATASET_NAME}/nas_{CONTROLLER_SAMPLES_PER_EPOCH}_samples_{CONTROLLER_SAMPLING_EPOCHS}_epochs_{FRAMES}_frames.json"
        self.initialize_analysis_file()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (CONTROLLER_INPUTS,1)
        print("IDs activos:", tf.keras.backend.get_uid())
        self.controller_model = self.create_controller_model(self.controller_input_shape, self.controller_batch_size)
        print("IDs activos:", tf.keras.backend.get_uid())
    def initialize_analysis_file(self):
        """Inicializa el archivo JSON con una estructura válida"""
        initial_data = {
            "metadata": {
                "total_controller_epochs": self.controller_sampling_epochs,
                "samples_per_epoch": self.controller_samples_per_epoch,
                "controller_train_epochs": self.controller_train_epochs,
                "architecture_train_epochs": self.architecture_train_epochs,
                "start_time": str(datetime.now()),
                "dataset": DATASET_NAME,
                "last_update": str(datetime.now())
            },
            "controller_epochs": [],
            "architectures": [],
            "top_architectures": []
        }
        
        # Asegurarse de que el archivo existe y tiene contenido válido
        os.makedirs(f"{os.path.abspath(os.path.curdir)}/results/{DATASET_NAME}", exist_ok=True)
        try:
            with open(self.analysis_file, 'r') as f:
                json.load(f)  # Solo verificar que es válido
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.analysis_file, 'w') as f:
                json.dump(initial_data, f, indent=4)

    def update_analysis_file(self, epoch_data):
        """Actualiza el archivo JSON convirtiendo tipos numpy a nativos"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Convertir datos numpy a tipos nativos de Python
                self.convert_numpy_types(epoch_data)
                
                # Leer datos existentes
                try:
                    with open(self.analysis_file, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {
                        "metadata": self.get_metadata(),
                        "controller_epochs": [],
                        "architectures": [],
                        "top_architectures": []
                    }
                
                # Actualizar datos
                data["controller_epochs"].append(epoch_data)
                data["architectures"].extend(epoch_data["sequences"])
                data["top_architectures"] = sorted(
                    data["architectures"],
                    key=lambda x: x["testing"]["accuracy"],
                    reverse=True
                )[:TOP_N]
                data["metadata"]["last_update"] = str(datetime.now())
                
                # Escribir con el encoder personalizado
                temp_file = self.analysis_file + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=4, cls=NumpyEncoder)
                
                os.replace(temp_file, self.analysis_file)
                break
                
            except Exception as e:
                print(f"Intento {attempt + 1} fallido: {str(e)}")
                if attempt == max_retries - 1:
                    raise RuntimeError("No se pudo actualizar el archivo de análisis")

    def convert_numpy_types(self, data):
        """Convierte recursivamente tipos numpy a tipos nativos de Python"""
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self.convert_numpy_types(value)
        elif isinstance(data, (list, tuple)):
            return [self.convert_numpy_types(item) for item in data]
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.float_, np.float16, np.float32, 
                            np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def get_metadata(self):
        """Devuelve metadatos con tipos compatibles con JSON"""
        return {
            "total_controller_epochs": int(self.controller_sampling_epochs),
            "samples_per_epoch": int(self.controller_samples_per_epoch),
            "controller_train_epochs": int(self.controller_train_epochs),
            "architecture_train_epochs": int(self.architecture_train_epochs),
            "start_time": str(datetime.now()),
            "dataset": str(DATASET_NAME),
            "target_classes": int(TARGET_CLASSES)
        }


    def append_model_metrics(self, sequence, test_accuracy_LSTM, pred_accuracy=None):
        self.data.append([sequence, test_accuracy_LSTM])

        
    def prepare_controller_data(self, sequences):
        # Secuencias de entrada (excluyendo el último token)
        xc = pad_sequences(sequences, maxlen=self.max_len, padding='post')[:, :-1]  # (N, 8)
        xc = xc.reshape(-1, self.max_len-1, 1)  # (N, 8, 1)

        # Targets (siguiente token para cada posición)
        yc = pad_sequences([seq[1:] for seq in sequences], maxlen=self.max_len-1, padding='post')  # (N, 8)
        yc = to_categorical(yc, num_classes=self.controller_classes)  # (N, 8, vocab_size)

        return xc, yc

    def get_discounted_reward(self, rewards):
        """
        Calcula discounted rewards para los tokens de CADA arquitectura.
        Input:
            rewards: Array de accuracy de las arquitecturas [R1, R2, R3, R4] (shape: (4,))
        Output:
            discounted_rewards: Array de forma (4, max_len) donde cada fila contiene 
                            los discounted rewards para los tokens de una arquitectura.
        """
        discounted_rewards = np.zeros((len(rewards), self.max_len), dtype=np.float32)
        
        for i, R in enumerate(rewards):
            # Calcula discounted reward para CADA token de la arquitectura i
            for t in range(self.max_len):
                discounted_rewards[i, t] = R * (self.controller_loss_alpha ** t)  # Gamma^t * R
        
        # Normalización por arquitectura (opcional)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=1, keepdims=True)) / \
                            (discounted_rewards.std(axis=1, keepdims=True) + 1e-10)
        
        return discounted_rewards

    def custom_loss(self, target, output):
        # 1. Calcular media móvil de los accuracys previos (baseline)
        if len(self.data) > 0:
            window_size = min(len(self.data), 10)  # Tamaño de ventana para la media móvil (ej. últimos 10)
            recent_accuracies = [item[1] for item in self.data[-window_size:]]
            self.baseline = np.mean(recent_accuracies)
        else:
            self.baseline = 0.5  # Valor inicial si no hay datos
        
        # 2. Obtener rewards (accuracy) y aplicar baseline
        self.rewards = np.array([item[1] - self.baseline for item in self.data[-self.controller_samples_per_epoch:]])  # (N,)
        
        # 3. Obtener discounted rewards POR TOKEN (nuevo formato: (N, max_len-1))
        self.discounted_rewards = self.get_discounted_reward(self.rewards)  # Ahora devuelve (N, max_len-1)
        
        # 4. Calcular pérdida para cada paso temporal
        loss = 0
        for i in range(self.max_len - 1):
            # Probabilidad de la acción tomada (shape: (N,))
            action_probs = K.sum(output[:, i, :] * target[:, i, :], axis=-1)
            
            # Pérdida para este paso (usando el discounted reward específico para cada token)
            loss += -K.log(action_probs + 1e-10) * self.discounted_rewards[:, i]  # (N,)
        
        return K.mean(loss)  # Promediar sobre batch y pasos temporales

    # def custom_loss(self, target, output):
    #     # 1. Obtener rewards (accuracy) y aplicar baseline
    #     baseline = 0.5
    #     rewards = np.array([item[1] - baseline for item in self.data[-self.controller_samples_per_epoch:]])  # (N,)
    #     # 2. Obtener discounted rewards POR TOKEN (nuevo formato: (N, max_len-1))
    #     self.discounted_rewards = self.get_discounted_reward(rewards)  # Ahora devuelve (N, max_len-1)
        
    #     # 3. Calcular pérdida para cada paso temporal
    #     loss = 0
    #     for i in range(self.max_len - 1):
    #         # Probabilidad de la acción tomada (shape: (N,))
    #         action_probs = K.sum(output[:, i, :] * target[:, i, :], axis=-1)
            
    #         # Pérdida para este paso (usando el discounted reward específico para cada token)
    #         loss += -K.log(action_probs + 1e-10) * self.discounted_rewards[:, i]  # (N,)
        
    #     return K.mean(loss)  # Promediar sobre batch y pasos temporales

    def LSTM_train(self, sequence):
        architecture = self.decode_sequence(sequence) 
        
        print('---------------------------------------LSTM_train')
        print('Architecture: ', architecture)
        print('Architecture - num_layers: ', architecture[0])
        print('Architecture - rnn: ', architecture[1])
        print('Architecture - units_0: ', architecture[2])
        print('Architecture - units_1: ', architecture[3])
        print('Architecture - units_2: ', architecture[4])
        print('Architecture - direction: ', architecture[5])
        print('Architecture - state: ', architecture[6])
        print('Architecture - seq: ', architecture[7])
        print('Architecture - cnn: ', architecture[8]) 
        cfg = config.Config(operation='train',rnn=architecture[1],direction=architecture[5],units=(architecture[2],architecture[3],architecture[4]),cnn=architecture[8],data=DATASET_NAME,frames=FRAMES,size=FRAME_SIZE,seq=architecture[7],state=architecture[6])               
        return cfg
       
    # def train_controller(self, controller_model, x, y):
    #     print(f"len self.data:{len(self.data)}")
    #     return self.train_controller(controller_model,
    #                              x,
    #                              y,
    #                              self.custom_loss,
    #                              self.controller_train_epochs)

    def search(self):
        historico_all_var = []
        historico_lote_var = []
        historico_all_mean = []
        historico_lote_mean = []
        historico_loss = [] 
        for controller_epoch in range(self.controller_sampling_epochs):
            epoch_data = {
                "epoch": controller_epoch,
                "timestamp": str(datetime.now()),
                "sequences": [],
                "controller_metrics": {
                    "loss": [],
                    "rewards": [],
                    "discounted_rewards": [],
                    "all_reward_mean":[],
                    "all_reward_var":[],
                    "sample_reward_mean":[],
                    "sample_reward_var":[],
                    "historico_loss":[]    
                }
            }
            print()
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')

            sequences = self.sample_architecture_sequences(self.controller_model, self.controller_samples_per_epoch)
            print('sequences: ', sequences)

            for i, sequence in enumerate(sequences):
                seq_data = {
                    "sequence": sequence,
                    "decoded_sequence": self.decode_sequence(sequence),
                    "training": {},
                    "testing": {}
                }
                print('controller epoch: {}'.format(controller_epoch))
                print('i: ', i)
                print('sequence: ', sequence)
                print('Architecture: ', self.decode_sequence(sequence))   

                cfg = self.LSTM_train(sequence)
                print_memory()
                test_accuracy_rnn = train_lstm(cfg)
                test_accuracy_rnn = float(test_accuracy_rnn.numpy()) if tf.is_tensor(test_accuracy_rnn) else float(test_accuracy_rnn)
                print("test_accuracy_rnn: ", test_accuracy_rnn)
                print_memory()
                self.append_model_metrics(sequence, test_accuracy_rnn)
                seq_data["testing"]["accuracy"] = test_accuracy_rnn
                epoch_data["sequences"].append(seq_data)
            xc, yc = self.prepare_controller_data(sequences)
            print("IDs activos:", tf.keras.backend.get_uid())
            print_memory()
            history = self.train_controller(self.controller_model,xc,yc,self.custom_loss,self.controller_train_epochs)
            print("IDs activos:", tf.keras.backend.get_uid())
            print_memory()
            self.controller_losses.append(history.history['loss'])
            all_controller_rewards = [item[1] for item in self.data]
            all_reward_var = np.var(all_controller_rewards,ddof=0)
            all_reward_mean = np.mean(all_controller_rewards)
            lote_controller_reward = [item[1] for item in self.data[-self.controller_samples_per_epoch:]]
            lote_reward_var = np.var(lote_controller_reward,ddof=0)
            lote_reward_mean = np.mean(lote_controller_reward)
            historico_all_var.append(all_reward_var)
            historico_lote_var.append(lote_reward_var)
            historico_all_mean.append(all_reward_mean)
            historico_lote_mean.append(lote_reward_mean)
            historico_loss.append(history.history['loss'][-1])
            print("\n" + "="*50)
            print("📊 RESUMEN DE REWARDS (CONTROLADOR)")
            print("="*50)
            print(f"► Baseline (media móvil): {self.baseline:.4f}")
            print(f"► HISTÓRICOS (n={len(all_controller_rewards)}):")
            print(f"   • Media (μ): {all_reward_mean:.4f}")
            print(f"   • Varianza (σ²): {all_reward_var:.4f}")
            print("-"*50)
            print(f"► LOTE ACTUAL (n={len(lote_controller_reward)}):")
            print(f"   • Media (μ): {lote_reward_mean:.4f}")
            print(f"   • Varianza (σ²): {lote_reward_var:.4f}")
            print("="*50 + "\n")
            print(self.controller_losses)
            print(f"########################################################### CONTROLLER LOSSES ######################################################################################################### ")
            print("\n" + "="*50)
            print("📊 Lista historicos var (CONTROLADOR)")
            print(historico_all_var)
            print(historico_lote_var)
            print("="*50 + "\n")
            print("\n" + "="*50)
            print("📊 Lista historicos mean (CONTROLADOR)")
            print(historico_all_mean)
            print(historico_lote_mean)
            print("="*50 + "\n")

            epoch_data["controller_metrics"].update({
                "loss": history.history['loss'],
                "rewards":self.rewards,
                "discounted_rewards": self.discounted_rewards,
                "all_reward_mean":historico_all_mean,
                "all_reward_var":historico_all_var,
                "sample_reward_mean":historico_lote_mean,
                "sample_reward_var":historico_lote_var,
                "historico_loss":historico_loss 
            })
            # Actualizar el archivo JSON después de cada época
            self.update_analysis_file(epoch_data)
            print_memory()
        # with open(self.nas_data_log, 'wb') as f:
        #     pickle.dump(self.data, f)
        # log_event()
        return self.data
