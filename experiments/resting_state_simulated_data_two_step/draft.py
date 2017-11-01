import sys
# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    sys.path.append('dcm_rnn')
elif '/share/apps/python3/' in sys.executable:
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
from scipy.interpolate import interp1d
import importlib
from scipy.fftpack import idct, dct
import math as mth
import scipy.io as sio

sys.path.append(os.path.join(PROJECT_DIR, 'experiments', 'resting_state_simulated_data'))

du = tb.DataUnit()





du = tb.DataUnit()
du._secured_data['if_random_neural_parameter'] = False
du._secured_data['if_random_hemodynamic_parameter'] = False
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False
du._secured_data['if_random_stimuli'] = True
du._secured_data['if_random_node_number'] = False
du._secured_data['if_random_stimuli_number'] = False
du._secured_data['if_random_delta_t'] = False
du._secured_data['if_random_scan_time'] = False
du._secured_data['if_x_nonlinearity'] = False
du._secured_data['if_resting_state'] = True


du._secured_data['t_delta'] = 1. / 64.
du._secured_data['t_scan'] = 5 * 60
du._secured_data['n_node'] = 3
du._secured_data['n_stimuli'] = 2

du._secured_data['A'] = np.array([[-0.8, 0, 0],
                                      [0, -0.8, 0],
                                      [0.4, -0.4, -0.8]])
du._secured_data['B'] = [np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   ]),
                         np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   ])]
du._secured_data['C'] = np.array([[0.15, 0], [0, 0.15], [0, 0]]).reshape(du.get('n_node'), du.get('n_stimuli'))


du.complete_data_unit(if_show_message=False, if_check_property=False)

plt.figure(0)
plt.plot(du.get('u'))
plt.figure(1)
plt.plot(du.get('x'))
plt.figure(2)
plt.plot(du.get('y'))

