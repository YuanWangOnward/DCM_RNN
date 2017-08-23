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
from scipy import signal

#  simulate data
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
du._secured_data['t_delta'] = 1. / 16.
du._secured_data['t_scan'] = 5 * 60
du._secured_data['n_node'] = 3
du._secured_data['n_stimuli'] = 2
du._secured_data['A'] = np.array([[-0.8, -0.4, 0],
                                  [0.4, -0.8, -0.4],
                                  [0, 0.2, -0.8]])
du._secured_data['B'] = [np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]]),
                         np.array([[0, 0, 0],
                                   [0.4, 0, 0],
                                   [0, 0, 0]])]
du._secured_data['C'] = np.array([[0.8, 0, 0], [0, 0, 0]]).reshape(3, 2)

h_parameter = du.get_standard_hemodynamic_parameters(du.get('n_node'))
h_parameter['x_h_coupling'] = 1.
h_parameter['TE'] = 0.04
h_parameter['V0'] = 4
h_parameter['V0'] = 0.4

du._secured_data['hemodynamic_parameter'] = h_parameter

du.u_t_low = 5
du.u_t_high = 5
du.u_interval_t_low = 10
du.u_interval_t_high = 25

du.complete_data_unit(if_show_message=True, if_check_property=False)

# plt.plot(du.get('u'))
# plt.plot(du.get('y'))

# specify initialization values, loss weighting factors, and mask (support of effective connectivity)
x_parameter_initial = {}
x_parameter_initial['A'] = - np.eye(du.get('n_node'))
x_parameter_initial['B'] = [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))]
x_parameter_initial['C'] = np.zeros((du.get('n_node'), du.get('n_stimuli')))
x_parameter_initial['C'][0, 0] = 1

h_parameter_inital = du.get_standard_hemodynamic_parameters(du.get('n_node'))
# h_parameter_inital['x_h_coupling'] = 1.

loss_weighting = {'prediction': 1., 'sparsity': 0., 'prior': 1., 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}

mask = {
    'Wxx': np.ones((du.get('n_node'), du.get('n_node'))),
    'Wxxu': [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))],
    'Wxu': np.zeros((du.get('n_node'), du.get('n_stimuli')))
}
mask['Wxx'][0, 2] = 0
mask['Wxx'][2, 0] = 0
mask['Wxxu'][1][1, 0] = 1
mask['Wxu'][0, 0] = 1

# resample setting
down_sample_rate = 32.

# covert to DCM needed by SPM
DCM = {}
DCM['a'] = mask['Wxx']
DCM['b'] = np.stack(mask['Wxxu'], axis=2)
DCM['c'] = mask['Wxu']
DCM['v'] = float(du.get('n_time_point') / down_sample_rate)
DCM['n'] = float(du.get('n_node'))
DCM['TE'] = du.get('hemodynamic_parameter').iloc[0]['TE']
DCM['delays'] = np.zeros(3)
DCM['down_sample_rate'] = down_sample_rate

U = {}
U['dt'] = float(du.get('t_delta') * down_sample_rate)
U['name'] = ['input_0', 'input_1']
U['u'] = du.get('u')[::int(down_sample_rate), :]
DCM['U'] = U

Y = {}
Y['dt'] = float(du.get('t_delta') * down_sample_rate)
# Y['X0'] = 0.
Y['y'] = du.get('y')[::int(down_sample_rate), :]
Y['name'] = ['node_0', 'node_1', 'node_2']
Y['Q'] = []
DCM['Y'] = Y

options = {}
options['nonlinear'] = 0.
options['two_state'] = 0.
options['stochastic'] = 0.
options['centre'] = 0.
options['induced'] = 0.
DCM['options'] = options

DCM['du'] = du

# pickle.dump(DCM, open('/Users/yuanwang/Desktop/DCM.pkl', 'wb'))

import scipy.io
scipy.io.savemat('/Users/yuanwang/Desktop/DCM.mat', mdict={'DCM': DCM})
