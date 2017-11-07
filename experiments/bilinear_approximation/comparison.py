# estimate connectivity parameter
import sys

# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    sys.path.append('dcm_rnn')
elif '/home/yw1225' in sys.executable:
    PROJECT_DIR = '/home/yw1225/projects/DCM_RNN'
    print("It seems a remote run on NYU HPC")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    matplotlib.use('agg')
else:
    PROJECT_DIR = '.'
    print("Not sure executing machine. Make sure to set PROJECT_DIR properly.")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf
import tf_model as tfm
import toolboxes as tb
import numpy as np
import os
import pickle
import datetime
import warnings
import sys
import random
import training_manager
import multiprocessing
from multiprocessing.pool import Pool
import itertools
import copy
import pandas as pd
import math as mth
from scipy.interpolate import interp1d
import scipy.io

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'bilinear_approximation')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
SAVE_PATH = os.path.join(EXPERIMENT_PATH, 'results')


TOTAL_BASE_NUMBER = 20
SAMPLE_PER_BASE = 500

# pool = Pool(os.cpu_count())
# pool.map(build_a_base, range(TOTAL_BASE_NUMBER))

current_base_number = 0

data_path = DATA_PATH + '/DB' + str(current_base_number) + '.pkl'
current_data = []

while len(current_data) < SAMPLE_PER_BASE:
    # new and setting
    du = tb.DataUnit()
    du._secured_data['if_random_node_number'] = True
    du._secured_data['if_random_stimuli'] = True
    du._secured_data['if_random_x_state_initial'] = False
    du._secured_data['if_random_h_state_initial'] = False
    du._secured_data['t_delta'] = 0.25
    du._secured_data['t_scan'] = 5 * 60
    du.u_amplitude = 0.1
    du.complete_data_unit(if_show_message=False, if_check_property=False)

    # add cores
    du_core = du.collect_parameter_core()
    current_data.append(du_core)
    print('Number of current cores base ' + str(len(current_data)))

# sort cores
current_data = sorted(current_data, key=lambda x: x.get('n_node'))

# save cores
print('Number of current cores base ' + str(len(current_data)))
with open(data_path, 'wb') as f:
    pickle.dump(current_data, f)

