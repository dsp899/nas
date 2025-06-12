import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import keras.backend as K

from controller import Controller
from CONSTANTS import *
from utils import clean_log
from run_rnn import train_lstm
import config

class NASLogger:
    """Manejador centralizado de logging con estadísticas completas"""
    
    def __init__(self, dataset_name, samples_per_epoch, sampling_epochs, frames):
        self.log_dir = f"{os.path.abspath(os.path.curdir)}/results/{dataset_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{self.log_dir}/nas_log_{timestamp}.json"
        self.initialize_log_file()
        
    def initialize_log_file(self):
        """Inicializa el archivo JSON con estructura completa para estadísticas"""
        initial_data = {
            "metadata": {
                "start_time": str(datetime.now()),
                "last_update": str(datetime.now()),
                "status": "running",
                "parameters": {
                    "controller_sampling_epochs": CONTROLLER_SAMPLING_EPOCHS,
                    "controller_samples_per_epoch": CONTROLLER_SAMPLES_PER_EPOCH,
                    "controller_train_epochs": CONTROLLER_TRAINING_EPOCHS,
                    "architecture_train_epochs": ARCHITECTURE_TRAINING_EPOCHS,
                    "controller_loss_alpha": CONTROLLER_LOSS_ALPHA,
                    "dataset": DATASET_NAME,
                    "frames": FRAMES
                }
            },
            "statistics": {
                "global": {
                    "accuracies": [],
                    "rewards": [],
                    "normalized_rewards": [],
                    "baselines": [],
                    "cumulative_stats": []  # Nuevo: estadísticas globales acumuladas por época
                },
                "per_epoch": []
            },
            "architectures": [],
            "best_architectures": []
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(initial_data, f, indent=4, cls=NumpyEncoder)

    def log_epoch(self, epoch_data):
        """Registra todos los datos de una época incluyendo estadísticas detalladas"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            # Actualizar listas globales
            data["statistics"]["global"]["accuracies"].extend(epoch_data["accuracies"])
            data["statistics"]["global"]["rewards"].extend(epoch_data["raw_rewards"])
            data["statistics"]["global"]["normalized_rewards"].extend(epoch_data["rewards"])
            data["statistics"]["global"]["baselines"].append(epoch_data["baseline"])
            
            # Calcular estadísticas globales acumuladas hasta esta época
            global_stats = {
                "epoch": epoch_data["epoch"],
                "timestamp": str(datetime.now()),
                "accuracies": self._calculate_stats(data["statistics"]["global"]["accuracies"]),
                "rewards": self._calculate_stats(data["statistics"]["global"]["rewards"]),
                "normalized_rewards": self._calculate_stats(data["statistics"]["global"]["normalized_rewards"]),
                "baseline": epoch_data["baseline"]
            }
            data["statistics"]["global"]["cumulative_stats"].append(global_stats)
            
            # Registrar datos de la época actual
            epoch_entry = {
                "epoch": epoch_data["epoch"],
                "timestamp": str(datetime.now()),
                "stats": {
                    "current_epoch": {
                        "accuracies": self._calculate_stats(epoch_data["accuracies"]),
                        "raw_rewards": self._calculate_stats(epoch_data["raw_rewards"]),
                        "normalized_rewards": self._calculate_stats(epoch_data["rewards"]),
                        "controller": {
                            "loss": epoch_data["controller_loss"],
                            "loss_stats": self._calculate_stats(epoch_data["controller_loss"]),
                            "final_loss": epoch_data["controller_loss"][-1]
                        }
                    },
                    "global_until_now": global_stats  # Incluir las globales acumuladas
                },
                "architectures": epoch_data["architectures"]
            }
            
            data["statistics"]["per_epoch"].append(epoch_entry)
            data["architectures"].extend(epoch_data["architectures"])
            
            # Actualizar mejores arquitecturas
            data["best_architectures"] = sorted(
                data["architectures"],
                key=lambda x: x["metrics"]["accuracy"],
                reverse=True
            )[:TOP_N]
            
            data["metadata"]["last_update"] = str(datetime.now())
            data["metadata"]["status"] = "running" if epoch_data["epoch"] < data["metadata"]["parameters"]["controller_sampling_epochs"] - 1 else "completed"
            
            # Escribir de forma atómica
            temp_file = self.log_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=4, cls=NumpyEncoder)
            
            os.replace(temp_file, self.log_file)
            
        except Exception as e:
            print(f"⚠️ Error logging epoch: {str(e)}")
            raise

    def _calculate_stats(self, data):
        """Helper para calcular estadísticas consistentes"""
        if len(data) == 0:
            return {}
        
        return {
            "values": data,
            "count": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q1": float(np.quantile(data, 0.25)),
            "q3": float(np.quantile(data, 0.75)),
            "last_10_mean": float(np.mean(data[-10:])) if len(data) >= 10 else float(np.mean(data))
        }

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para manejar tipos numpy en JSON"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class MYLSTMNAS(Controller):
    def __init__(self):
        # Configuración inicial
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.controller_samples_per_epoch = CONTROLLER_SAMPLES_PER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA
        
        # Datos y estado
        self.architectures_data = []  # (sequence, accuracy, raw_reward, normalized_reward)
        self.controller_losses = []
        self.baseline = 0.0
        self.min_reward = 0.0
        
        # Inicializar logger
        self.logger = NASLogger(DATASET_NAME, 
                              self.controller_samples_per_epoch, 
                              self.controller_sampling_epochs, 
                              FRAMES)
        
        super().__init__()
        self.controller_input_shape = (CONTROLLER_INPUTS, 1)
        self.controller_model = self.create_controller_model(self.controller_input_shape, 
                                                          self.controller_samples_per_epoch)
        clean_log()

        self.data_file = os.path.join(self.logger.log_dir, f"architectures_data_{DATASET_NAME}_{FRAMES}_Ev_{ARCHITECTURE_TRAINING_EPOCHS}_.json")
        self._load_existing_data()


    def _load_existing_data(self):
        """Carga arquitecturas ya evaluadas desde archivo si existe"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    raw_data = json.load(f)
                    self.architectures_data = [
                        (np.array(item["sequence"]), 
                         item["accuracy"], 
                         item["raw_reward"], 
                         item["normalized_reward"]) 
                        for item in raw_data
                    ]
                    print(f"✅ {len(self.architectures_data)} arquitecturas cargadas desde disco.")
            except Exception as e:
                print(f"⚠️ Error cargando arquitecturas previas: {str(e)}")

    def _save_data_to_disk(self):
        """Guarda las arquitecturas evaluadas en disco en formato JSON"""
        try:
            json_data = [
                {
                    "sequence": seq,
                    "accuracy": acc,
                    "raw_reward": raw,
                    "normalized_reward": norm
                }
                for seq, acc, raw, norm in self.architectures_data
            ]
            temp_file = self.data_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(json_data, f, indent=4, cls=NumpyEncoder)
            os.replace(temp_file, self.data_file)
        except Exception as e:
            print(f"⚠️ Error guardando arquitecturas: {str(e)}")



    def prepare_controller_data(self, sequences):
        """
        Prepara los datos para entrenar el controlador:
        1. Pad sequences para que todas tengan la misma longitud
        2. Crea inputs (todos los tokens excepto el último)
        3. Crea targets (todos los tokens excepto el primero)
        """
        # Padding de secuencias
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # Inputs: todos los tokens excepto el último
        xc = padded_sequences[:, :-1]  # (batch_size, max_len-1)
        xc = np.expand_dims(xc, axis=-1)  # (batch_size, max_len-1, 1)
        
        # Targets: todos los tokens excepto el primero (one-hot encoded)
        yc = pad_sequences([seq[1:] for seq in sequences], 
                          maxlen=self.max_len-1, 
                          padding='post')  # (batch_size, max_len-1)
        yc = to_categorical(yc, num_classes=self.controller_classes)  # (batch_size, max_len-1, vocab_size)
        
        return xc, yc

    def _calculate_reward(self, accuracy):
        """Calcula el reward de forma robusta"""
        raw_reward = accuracy - self.baseline
        normalized_reward = raw_reward / (self.baseline + 1e-10)
        normalized_reward = max(normalized_reward, self.min_reward)
        return raw_reward, normalized_reward

    def _update_baseline(self, new_accuracies):
        """Actualiza el baseline usando una media móvil exponencial"""
        if not hasattr(self, '_ema_alpha'):
            self._ema_alpha = 0.3
        
        current_mean = np.mean(new_accuracies)
        
        if not hasattr(self, '_ema_baseline'):
            self._ema_baseline = current_mean
        else:
            self._ema_baseline = (self._ema_alpha * current_mean + 
                                (1 - self._ema_alpha) * self._ema_baseline)
        
        self.baseline = min(self._ema_baseline, 0.95)
    #Con discounted reward
    def _get_discounted_rewards(self, rewards):
        """Calcula discounted rewards de forma robusta"""
        discounted = np.zeros((len(rewards), self.max_len), dtype=np.float32)
        
        for i, R in enumerate(rewards):
            if R > 0:
                for t in range(self.max_len):
                    discounted[i, t] = R * (self.controller_loss_alpha ** t)
            else:
                discounted[i, :] = 0.1 * (self.controller_loss_alpha ** np.arange(self.max_len))
        
        # Normalización robusta
        mean = discounted.mean(axis=1, keepdims=True)
        std = discounted.std(axis=1, keepdims=True) + 1e-10
        
        normalized = (discounted - mean) / std
        
        if np.all(normalized == 0):
            normalized = np.ones_like(normalized) * 0.1
        
        return normalized
    # sin discounted, todos los token tiene la misma importancia dentro de la secuencia
    # def _get_discounted_rewards(self, rewards):
    #     """Calcula rewards uniformes para todos los tokens (sin jerarquía ni descuento)"""
    #     discounted = np.zeros((len(rewards), self.max_len), dtype=np.float32)
        
    #     for i, R in enumerate(rewards):
    #         if R > 0:
    #             # Asignar el mismo reward R a todos los tokens en la secuencia
    #             discounted[i, :] = R
    #         else:
    #             # En caso de reward negativo o cero, asignar un valor pequeño uniforme
    #             discounted[i, :] = 0.1
        
    #     # Normalización robusta por secuencia (opcional, pero recomendable)
    #     mean = discounted.mean(axis=1, keepdims=True)
    #     std = discounted.std(axis=1, keepdims=True) + 1e-10
        
    #     normalized = (discounted - mean) / std
        
    #     # Evitar que todos los valores sean cero tras normalización
    #     if np.all(normalized == 0):
    #         normalized = np.ones_like(normalized) * 0.1
        
    #     return normalized

    def _print_architecture_report(self, seq_idx, sequence, accuracy, raw_reward, norm_reward):
        """Reporte detallado para cada arquitectura evaluada"""
        arch = self.decode_sequence(sequence)
        print(f"\n🏗️ Architecture #{seq_idx + 1} Report:")
        print("="*50)
        print(f"🔧 Structure: {arch}")
        print(f"📊 Accuracy:  {accuracy:.4f}")
        print(f"🏆 Raw Reward: {raw_reward:.4f} (baseline: {self.baseline:.4f})")
        print(f"🎯 Normalized Reward: {norm_reward:.4f}")
        print("="*50)

    def _print_epoch_summary(self, epoch, acc_stats, reward_stats, norm_reward_stats, loss_stats):
        """Reporte completo de la época"""
        print("\n" + "="*80)
        print(f"📈 EPOCH {epoch} SUMMARY REPORT")
        print("="*80)
        
        print(f"\n🧮 Baseline: {self.baseline:.4f}")
        
        print("\n🔍 Accuracy Statistics:")
        print(f"   • Mean:   {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}")
        print(f"   • Range:  [{acc_stats['min']:.4f}, {acc_stats['max']:.4f}]")
        print(f"   • Median: {acc_stats['median']:.4f}")
        
        print("\n🏆 Reward Statistics (Raw):")
        print(f"   • Mean:   {reward_stats['mean']:.4f} ± {reward_stats['std']:.4f}")
        print(f"   • Range:  [{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]")
        
        print("\n🎯 Reward Statistics (Normalized):")
        print(f"   • Mean:   {norm_reward_stats['mean']:.4f} ± {norm_reward_stats['std']:.4f}")
        print(f"   • Range:  [{norm_reward_stats['min']:.4f}, {norm_reward_stats['max']:.4f}]")
        
        print("\n🧠 Controller Training:")
        print(f"   • Final Loss: {loss_stats['final_loss']:.4f}")
        print(f"   • Loss Trend: {loss_stats['min']:.4f} → {loss_stats['max']:.4f}")
        print("="*80 + "\n")

    def train_architecture(self, sequence):
        """Entrena y evalúa una arquitectura LSTM específica"""
        arch = self.decode_sequence(sequence)
        
        cfg = config.Config(
            operation='train',
            rnn=arch[1],
            direction=arch[5],
            units=(arch[2], arch[3], arch[4]),
            cnn=arch[8],
            data=DATASET_NAME,
            frames=FRAMES,
            size=FRAME_SIZE,
            seq=arch[7],
            state=arch[6]
        )
        # cfg = config.Config(
        #     operation='train',
        #     rnn=arch[3],
        #     direction=arch[7],
        #     units=(arch[4], arch[5], arch[6]),
        #     cnn=arch[1],
        #     data=DATASET_NAME,
        #     frames=FRAMES,
        #     size=FRAME_SIZE,
        #     seq=arch[0],
        #     state=arch[8]
        # )
        
        accuracy = train_lstm(cfg)
        accuracy = float(accuracy.numpy()) if tf.is_tensor(accuracy) else float(accuracy)
        
        raw_reward, norm_reward = self._calculate_reward(accuracy)
        
        return accuracy, raw_reward, norm_reward

    def custom_loss(self, target, output):
        """Función de pérdida personalizada usando rewards normalizados"""
        recent_data = self.architectures_data[-self.controller_samples_per_epoch:]
        recent_rewards = np.array([item[3] for item in recent_data])
        
        self.discounted_rewards = self._get_discounted_rewards(recent_rewards)
        
        loss = 0
        for i in range(self.max_len - 1):
            action_probs = K.sum(output[:, i, :] * target[:, i, :], axis=-1)
            loss += -K.log(action_probs + 1e-10) * self.discounted_rewards[:, i]
        
        return K.mean(loss)

    def search(self):
        """Búsqueda NAS principal con manejo robusto de rewards"""
        for epoch in range(self.controller_sampling_epochs):
            epoch_data = {
                "epoch": epoch,
                "accuracies": [],
                "raw_rewards": [],
                "rewards": [],
                "baseline": self.baseline,
                "controller_loss": [],
                "architectures": []
            }
            
            print(f"\n🚀 Starting Epoch {epoch + 1}/{self.controller_sampling_epochs}")
            print(f"📌 Current Baseline: {self.baseline:.4f}")
            
            # 1. Sample architectures
            sequences = self.sample_architecture_sequences(
                self.controller_model, 
                self.controller_samples_per_epoch
            )
            
            # 2. Train and evaluate architectures
            for i, seq in enumerate(sequences):
                print(f"🔧 Structure: {self.decode_sequence(seq)}")

                # Revisar si ya existe esta secuencia en self.architectures_data
                if any(np.array_equal(self.decode_sequence(seq), s[0]) for s in self.architectures_data):
                    print(f"⚠️ Secuencia ya evaluada, se omite entrenamiento: {self.decode_sequence(seq)}")
                    existing = next(s for s in self.architectures_data if np.array_equal(self.decode_sequence(seq), s[0]))
                    accuracy, raw_reward, norm_reward = existing[1], existing[2], existing[3]
                else:
                    accuracy, raw_reward, norm_reward = self.train_architecture(seq)
                    self.architectures_data.append((self.decode_sequence(seq), accuracy, raw_reward, norm_reward))
                    self._save_data_to_disk()
                # accuracy, raw_reward, norm_reward = self.train_architecture(seq)
                # self.architectures_data.append((seq, accuracy, raw_reward, norm_reward))
                
                epoch_data["accuracies"].append(accuracy)
                epoch_data["raw_rewards"].append(raw_reward)
                epoch_data["rewards"].append(norm_reward)
                
                arch_details = {
                    "sequence": seq,
                    "decoded": self.decode_sequence(seq),
                    "metrics": {
                        "accuracy": accuracy,
                        "raw_reward": raw_reward,
                        "normalized_reward": norm_reward,
                        "baseline": self.baseline
                    },
                    "timestamp": str(datetime.now())
                }
                epoch_data["architectures"].append(arch_details)
                
                self._print_architecture_report(i, seq, accuracy, raw_reward, norm_reward)
            
            # 3. Update baseline
            self._update_baseline(epoch_data["accuracies"])
            epoch_data["baseline"] = self.baseline
            
            # 4. Train controller
            xc, yc = self.prepare_controller_data(sequences)
            history = self.train_controller(
                self.controller_model,
                xc,
                yc,
                self.custom_loss,
                self.controller_train_epochs
            )
            
            epoch_data["controller_loss"] = history.history['loss']
            self.controller_losses.append(history.history['loss'])
            
            # 5. Calcular estadísticas
            acc_stats = self.logger._calculate_stats(epoch_data["accuracies"])
            reward_stats = self.logger._calculate_stats(epoch_data["raw_rewards"])
            norm_reward_stats = self.logger._calculate_stats(epoch_data["rewards"])
            loss_stats = self.logger._calculate_stats(epoch_data["controller_loss"])
            loss_stats["final_loss"] = epoch_data["controller_loss"][-1]
            
            # 6. Imprimir y guardar reporte
            self._print_epoch_summary(epoch, acc_stats, reward_stats, norm_reward_stats, loss_stats)
            self.logger.log_epoch(epoch_data)
        
        print("\n🎉 NAS Search Completed!")
        print(f"📂 Full log saved to: {self.logger.log_file}")
        
        return {
            "architectures": self.architectures_data,
            "log_file": self.logger.log_file
        }