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

class EnhancedNASLogger:
    """Enhanced logger with reward tracking and moving mean baseline"""
    
    def __init__(self, dataset_name):
        self.log_dir = f"{os.path.abspath(os.path.curdir)}/results/{dataset_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{self.log_dir}/nas_log_{timestamp}.json"
        self.initialize_log_file()
        
    def initialize_log_file(self):
        """Initialize log file with enhanced structure"""
        initial_data = {
            "metadata": {
                "start_time": str(datetime.now()),
                "last_update": str(datetime.now()),
                "epoch_count": 0,
                "moving_baseline": None,
                "parameters": {
                    "controller_sampling_epochs": CONTROLLER_SAMPLING_EPOCHS,
                    "controller_samples_per_epoch": CONTROLLER_SAMPLES_PER_EPOCH,
                    "controller_train_epochs": CONTROLLER_TRAINING_EPOCHS,
                    "architecture_train_epochs": ARCHITECTURE_TRAINING_EPOCHS,
                    "dataset": DATASET_NAME,
                    "frames": FRAMES,
                    "baseline_alpha": 0.2  # Default value
                }
            },
            "accuracies": [],
            "rewards": [],  # New: stores calculated rewards per architecture
            "losses": [],
            "architectures": []
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(initial_data, f, indent=4, cls=NumpyEncoder)

    def log_epoch(self, epoch, accuracies, rewards, losses, architectures):
        """Log epoch data with reward tracking"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            # Update metadata
            data["metadata"]["last_update"] = str(datetime.now())
            data["metadata"]["epoch_count"] = epoch + 1
            
            # Append new data
            data["accuracies"].extend(accuracies)
            data["rewards"].extend(rewards)  # Store rewards
            data["losses"].extend(losses)
            
            # Add architecture details with their rewards
            for arch, reward in zip(architectures, rewards):
                arch["reward"] = float(reward)  # Store reward with architecture
                data["architectures"].append(arch)
            
            # Write atomically
            temp_file = self.log_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=4, cls=NumpyEncoder)
            
            os.replace(temp_file, self.log_file)
            
        except Exception as e:
            print(f"Error logging epoch: {str(e)}")
            raise

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types in JSON"""
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
        # Configuration
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.controller_samples_per_epoch = CONTROLLER_SAMPLES_PER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        
        # Reward calculation parameters
        self.reward_baseline = None
        self.baseline_alpha = 0.2  # EMA smoothing factor
        self.reward_scale = 1.0    # Reward scaling factor
        
        # Data and state
        self.architectures_data = []  # Now stores (sequence, accuracy, reward)
        self.controller_losses = []
        
        # Initialize enhanced logger
        self.logger = EnhancedNASLogger(DATASET_NAME)
        
        super().__init__()
        self.controller_input_shape = (CONTROLLER_INPUTS, 1)
        self.controller_model = self.create_controller_model(self.controller_input_shape, 
                                                          self.controller_samples_per_epoch)
        clean_log()

        self.data_file = os.path.join(self.logger.log_dir, f"architectures_data_{DATASET_NAME}_{FRAMES}_Ev_{ARCHITECTURE_TRAINING_EPOCHS}_.json")
        self._load_existing_data()

    def _load_existing_data(self):
        """Load previously evaluated architectures with rewards"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    raw_data = json.load(f)
                    self.architectures_data = [
                        (np.array(item["sequence"]), 
                        item["accuracy"],
                        item.get("reward", 0.0))  # Load reward if exists
                        for item in raw_data
                    ]
                    print(f"✅ {len(self.architectures_data)} architectures loaded from disk.")
            except Exception as e:
                print(f"⚠️ Error loading previous architectures: {str(e)}")

    def _save_data_to_disk(self):
        """Save evaluated architectures with rewards"""
        try:
            json_data = [
                {
                    "sequence": seq,
                    "accuracy": acc,
                    "reward": rew  # Store reward
                }
                for seq, acc, rew in self.architectures_data
            ]
            temp_file = self.data_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(json_data, f, indent=4, cls=NumpyEncoder)
            os.replace(temp_file, self.data_file)
        except Exception as e:
            print(f"⚠️ Error saving architectures: {str(e)}")

    def _print_architecture_report(self, seq_idx, sequence, accuracy):
        """Report for each evaluated architecture"""
        arch = self.decode_sequence(sequence)
        print(f"\n🏗️ Architecture #{seq_idx + 1} Report:")
        print("="*50)
        print(f"🔧 Structure: {arch}")
        print(f"📊 Accuracy:  {accuracy:.4f}")
        print("="*50)

    def _print_epoch_summary(self, epoch, acc_stats, loss_stats):
        """Epoch summary report"""
        print("\n" + "="*80)
        print(f"📈 EPOCH {epoch} SUMMARY REPORT")
        print("="*80)
        
        print("\n🔍 Accuracy Statistics:")
        print(f"   • Mean:   {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}")
        print(f"   • Range:  [{acc_stats['min']:.4f}, {acc_stats['max']:.4f}]")
        print(f"   • Median: {acc_stats['median']:.4f}")
        
        print("\n🧠 Controller Training:")
        print(f"   • Final Loss: {loss_stats['final_loss']:.4f}")
        print(f"   • Loss Trend: {loss_stats['min']:.4f} → {loss_stats['max']:.4f}")
        print("="*80 + "\n")

    def _update_baseline(self, accuracies):
        """Update moving baseline with Exponential Moving Average"""
        current_mean = np.mean(accuracies)
        
        if self.reward_baseline is None:
            self.reward_baseline = current_mean
        else:
            self.reward_baseline = (self.baseline_alpha * current_mean + 
                                  (1 - self.baseline_alpha) * self.reward_baseline)
        
        # Dynamic adjustment of baseline alpha
        if np.std(accuracies) < 0.02:  # Low variance
            self.baseline_alpha = min(0.3, self.baseline_alpha + 0.01)
        else:
            self.baseline_alpha = max(0.1, self.baseline_alpha - 0.005)

    def _calculate_rewards(self, accuracies):
        """Calculate rewards using moving baseline and proper scaling"""
        # Update baseline first
        self._update_baseline(accuracies)
        
        # Center rewards around baseline
        centered = np.array(accuracies) - self.reward_baseline
        
        # Scale rewards to have unit variance (stabilizes learning)
        std = np.std(centered) + 1e-8
        scaled_rewards = centered / std
        
        # Optional: Apply tanh to keep rewards in reasonable range
        rewards = np.tanh(scaled_rewards * self.reward_scale)
        
        return rewards

    def _get_token_rewards(self, rewards):
        """Expand architecture rewards to token-level rewards"""
        token_rewards = np.zeros((len(rewards), self.max_len), dtype=np.float32)
        token_rewards[:] = rewards[:, np.newaxis]  # Same reward for all tokens
        return token_rewards

    def prepare_controller_data(self, sequences):
        """Prepare data for controller training"""
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        xc = padded_sequences[:, :-1]
        xc = np.expand_dims(xc, axis=-1)
        yc = pad_sequences([seq[1:] for seq in sequences], 
                          maxlen=self.max_len-1, 
                          padding='post')
        yc = to_categorical(yc, num_classes=self.controller_classes)
        return xc, yc

    def custom_loss(self, target, output):
        """Improved loss function with moving baseline rewards"""
        recent_data = self.architectures_data[-self.controller_samples_per_epoch:]
        recent_accuracies = np.array([item[1] for item in recent_data])
        
        # Calculate rewards
        rewards = self._calculate_rewards(recent_accuracies)
        token_rewards = self._get_token_rewards(rewards)
        
        # Calculate loss
        loss = 0
        for i in range(self.max_len - 1):
            action_probs = K.sum(output[:, i, :] * target[:, i, :], axis=-1)
            loss += -K.log(action_probs + 1e-10) * token_rewards[:, i]
        
        # Add entropy regularization
        entropy = -K.sum(output * K.log(output + 1e-10), axis=-1)
        entropy_bonus = 0.01 * K.mean(entropy)
        
        return K.mean(loss) - entropy_bonus

    def train_architecture(self, sequence):
        """Train and evaluate architecture, return accuracy"""
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
        
        accuracy = train_lstm(cfg)
        return float(accuracy.numpy()) if tf.is_tensor(accuracy) else float(accuracy)

    def search(self):
        """Main NAS search loop with enhanced reward tracking"""
        for epoch in range(self.controller_sampling_epochs):
            epoch_accuracies = []
            epoch_rewards = []
            epoch_losses = []
            epoch_architectures = []
            
            print(f"\n🚀 Starting Epoch {epoch + 1}/{self.controller_sampling_epochs}")
            if self.reward_baseline is not None:
                print(f"📊 Current Baseline: {self.reward_baseline:.4f}")
            
            # 1. Sample architectures
            sequences = self.sample_architecture_sequences(
                self.controller_model, 
                self.controller_samples_per_epoch
            )
            
            # 2. Train and evaluate architectures
            for i, seq in enumerate(sequences):
                print(f"🔧 Evaluating architecture {i+1}/{len(sequences)}")
                print(f"🔧 Structure: {self.decode_sequence(seq)}")
                
                # Check for existing evaluation
                existing = next((s for s in self.architectures_data if np.array_equal(seq, s[0])), None)
                if existing:
                    print("⚠️ Using previously evaluated architecture")
                    accuracy, reward = existing[1], existing[2]
                else:
                    accuracy = self.train_architecture(seq)
                    reward = 0.0  # Temporary placeholder
                    self.architectures_data.append((seq, accuracy, reward))
                
                epoch_accuracies.append(accuracy)
                
                arch_details = {
                    "sequence": seq,
                    "decoded": self.decode_sequence(seq),
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "timestamp": str(datetime.now())
                }
                epoch_architectures.append(arch_details)
                
                self._print_architecture_report(i, seq, accuracy)
            
            # Calculate rewards for all architectures in this epoch
            epoch_rewards = self._calculate_rewards(epoch_accuracies)
            
            # Update rewards in architectures_data
            start_idx = len(self.architectures_data) - len(epoch_accuracies)
            for i, reward in enumerate(epoch_rewards):
                seq, acc, _ = self.architectures_data[start_idx + i]
                self.architectures_data[start_idx + i] = (seq, acc, reward)
            
            self._save_data_to_disk()
            
            # 3. Train controller
            xc, yc = self.prepare_controller_data(sequences)
            history = self.train_controller(
                self.controller_model,
                xc,
                yc,
                self.custom_loss,
                self.controller_train_epochs
            )
            
            epoch_losses = history.history['loss']
            self.controller_losses.append(epoch_losses)
            
            # 4. Log and report
            acc_stats = {
                "mean": np.mean(epoch_accuracies),
                "std": np.std(epoch_accuracies),
                "min": np.min(epoch_accuracies),
                "max": np.max(epoch_accuracies),
                "median": np.median(epoch_accuracies)
            }
            
            reward_stats = {
                "mean": np.mean(epoch_rewards),
                "std": np.std(epoch_rewards),
                "min": np.min(epoch_rewards),
                "max": np.max(epoch_rewards)
            }
            
            loss_stats = {
                "mean": np.mean(epoch_losses),
                "std": np.std(epoch_losses),
                "min": np.min(epoch_losses),
                "max": np.max(epoch_losses),
                "final_loss": epoch_losses[-1]
            }
            
            # Enhanced reporting
            print(f"\n📈 Epoch {epoch} Reward Stats:")
            print(f"   • Baseline: {self.reward_baseline:.4f}")
            print(f"   • Rewards: [{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]")
            print(f"   • Mean Reward: {reward_stats['mean']:.4f} ± {reward_stats['std']:.4f}")
            
            self._print_epoch_summary(epoch, acc_stats, loss_stats)
            self.logger.log_epoch(epoch, epoch_accuracies, epoch_rewards, epoch_losses, epoch_architectures)
        
        print("\n🎉 NAS Search Completed!")
        print(f"📂 Full log saved to: {self.logger.log_file}")
        
        return {
            "architectures": self.architectures_data,
            "log_file": self.logger.log_file
        }