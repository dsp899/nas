
from utils import *

from myLSTMNAS import MYLSTMNAS

#import tensorflow.compat.v1 as tf

import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

#Version Info
print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
print('tf version: ', tf.__version__)
print('tf.keras version:', tf.keras.__version__)
#tf.config.experimental.enable_op_determinism
tf.keras.utils.set_random_seed(1337)

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
#tf.compat.v1.set_random_seed(264)

nas_object = MYLSTMNAS()
data = nas_object.search()
#get_top_n_architectures(TOP_N)
