########################################################
#                   NAS PARAMETERS                     #
########################################################
CONTROLLER_SAMPLING_EPOCHS = 100    # how many total controller epochs
CONTROLLER_SAMPLES_PER_EPOCH = 4   # how many architectures must be sampled in each controller epoch 
CONTROLLER_TRAINING_EPOCHS =5    # how many epochs to train controller on each controller epoch
ARCHITECTURE_TRAINING_EPOCHS = 1  # how many epochs to train each generated architecture
CONTROLLER_LOSS_ALPHA = 0.9        # the alpha value needed to calculate discounted reward

########################################################
#               CONTROLLER PARAMETERS                  #
########################################################
SEQUENCE_LENGTH = 9

CONTROLLER_LSTM_DIM = 50 
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.001
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0

########################################################
#                   DATA PARAMETERS                    #
########################################################
DATASET_NAME = 'pmi' # 'all' 'all50' 'pmi' 'pmi50'
FRAMES = 36
FRAME_SIZE = 299
########################################################
#                  OUTPUT PARAMETERS                   #
########################################################
TOP_N = CONTROLLER_SAMPLING_EPOCHS * CONTROLLER_SAMPLES_PER_EPOCH  # 5

CONTROLLER_INPUTS = SEQUENCE_LENGTH - 1 # all inputs and one output
CONTROLLER_OUTPUTS = 49                 # vocabulary length 
