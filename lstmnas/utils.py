import os
import shutil
import pickle
import numpy as np
from itertools import groupby
from matplotlib import pyplot as plt

from CONSTANTS import *

from LSTMSearchSpace import LSTMSearchSpace

########################################################
#                       LOGGING                        #
########################################################

def clean_log():
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))

def log_event():
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/event{}'.format(np.random.randint(10000))
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)

def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))


########################################################
#                 RESULTS PROCESSING                   #
########################################################


def load_nas_data():
    event = get_latest_event_id()
    data_file = 'LOGS/event{}/nas_data.pkl'.format(event)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def sort_search_data(nas_data):
    val_accs = [item[1] for item in nas_data]
    sorted_idx = np.argsort(val_accs)[::-1]
    nas_data = [nas_data[x] for x in sorted_idx]
    return nas_data

########################################################
#                EVALUATION AND PLOTS                  #
########################################################

def get_top_n_architectures(n):
    data = load_nas_data()
    print('data1 :', data)
    data = sort_search_data(data)
    print('data2 :', data)
    search_space = LSTMSearchSpace()
    print('Top {} Architectures:'.format(n))
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation Accuracy:', seq_data[1])
