# replot results
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
import scipy.io as sio
import matplotlib.ticker as mtick


N_NODE = 5
N_STIMULI = 3

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'simulation_example')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
RESULT_PATH = os.path.join(EXPERIMENT_PATH, 'results')
IMAGE_PATH = os.path.join(EXPERIMENT_PATH, 'images')
SAVE_NAME = os.path.join(RESULT_PATH, 'node_' + str(N_NODE) + '_stimuli_' + str(N_STIMULI) + '.pkl')


du = tb.DataUnit()
du._secured_data['if_random_node_number'] = False
du._secured_data['if_random_stimuli'] = True
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False
du._secured_data['n_node'] = N_NODE
du._secured_data['n_stimuli'] = N_STIMULI
du._secured_data['t_delta'] = 0.25
du._secured_data['t_scan'] = 5 * 60
du.complete_data_unit(if_show_message=False, if_check_property=True)
pickle.dump(du, open(SAVE_NAME, 'wb'))



x_axis = np.array(range(du.get('n_time_point'))) * du.get('t_delta')
matplotlib.rc('font', **{'size': 16})
plt.figure(figsize=(8, 3))
ax = plt.subplot(2, 1, 1)
plt.plot(x_axis, du.get('u'), alpha=1.)
ax.set_xticklabels([])
plt.ylabel('input')
plt.subplot(2, 1, 2)
plt.plot(x_axis, du.get('y'), alpha=1.)
plt.xlabel('time (second)')
plt.ylabel('fMRI')


