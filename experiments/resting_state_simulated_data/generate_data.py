# generate data for DCM-RNN and SPM estimation
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
import scipy as sp
import scipy.io

SIMULATION_X_NONLINEARITY = 'None'
EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'resting_state_simulated_data')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
CORE_PATH = os.path.join(DATA_PATH, 'core_' + SIMULATION_X_NONLINEARITY + '.pkl')
SAVE_PATH_PKL = os.path.join(DATA_PATH, 'du_DCM_RNN_' + SIMULATION_X_NONLINEARITY + '.pkl')
SAVE_PATH_MAT = os.path.join(DATA_PATH, 'DCM_initial_' + SIMULATION_X_NONLINEARITY + '.mat')
SNR = 3

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

du._secured_data['x_nonlinearity_type'] = SIMULATION_X_NONLINEARITY
du._secured_data['t_delta'] = 1. / 64.
du._secured_data['t_scan'] = 5 * 60
du._secured_data['n_node'] = 3
du._secured_data['n_stimuli'] = 3

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
                                   ]),
                         np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   ])]
du._secured_data['C'] = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]).reshape(du.get('n_node'), du.get('n_stimuli'))

# adjust hemodynamic parameter according to the ones used in SPM DCM
# with decay = 0 and transit = 0
# k = sd = H(1)*exp(decay) = 0.64
# gamma = H(2) = 0.32
# tao = tt = H(3)*exp(transit) = 2
# alpha = H(4) = 0.32
# E0 = H(5) = 0.4
# nu0 = theta0 = 40.3
# epsilon as in put P.epsilon, epsilon = exp(P.epsilon)
hemodynamic_parameter = du.get_standard_hemodynamic_parameters(du.get('n_node'))
hemodynamic_parameter['alpha'] = 0.32
hemodynamic_parameter['E0'] = 0.4
hemodynamic_parameter['k'] = 0.64
hemodynamic_parameter['gamma'] = 0.32
hemodynamic_parameter['tao'] = 2.

hemodynamic_parameter['epsilon'] = 0.4
hemodynamic_parameter['V0'] = 4.
hemodynamic_parameter['TE'] = 0.04
hemodynamic_parameter['r0'] = 25.
hemodynamic_parameter['theta0'] = 40.3
hemodynamic_parameter['x_h_coupling'] = 1.
du._secured_data['hemodynamic_parameter'] = hemodynamic_parameter

# scan
du.complete_data_unit(if_show_message=False, if_check_property=False)
for i in range(du.get('n_node') + 1):
    plt.subplot(4, 1, i + 1)
    if i == 0:
        plt.plot(du.get('x'))
    else:
        plt.plot(du.get('y')[:, i - 1])

# add noise
noise_std = np.sqrt(np.var(du.get('y').flatten())) / SNR
noise = np.random.randn(du.get('y').shape[0], du.get('y').shape[1]) * noise_std
du._secured_data['y_noised'] = du._secured_data['y'] + noise
for i in range(du.get('n_node') + 1):
    plt.subplot(4, 1, i + 1)
    if i == 0:
        plt.plot(du.get('u'))
    else:
        plt.plot(du.get('y_noised')[:, i - 1])

# save data after finding a good u
core = du.collect_parameter_core('y_noised')
pickle.dump(core, open(CORE_PATH, 'wb'))

## do down sample and up sample, make the data used for DCM-RNN inference
# load data
core = tb.load_template(CORE_PATH)
du_original = tb.DataUnit()
du_original.load_parameter_core(core)
du_original.recover_data_unit()
du = du_original.resample_data_unit()
pickle.dump(du, open(SAVE_PATH_PKL, 'wb'))


# create DCM structure for SPM DCM
core = tb.load_template(CORE_PATH)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()

mask = du.create_support_mask()
down_sample_rate_u = 4
down_sample_rate_y = 128
# covert to DCM needed by SPM
DCM = {}
DCM['a'] = mask['Wxx']
DCM['b'] = np.stack(mask['Wxxu'], axis=2)
DCM['c'] = mask['Wxu']
DCM['v'] = float(du.get('n_time_point') / down_sample_rate_y)
DCM['n'] = float(du.get('n_node'))
DCM['TE'] = du.get('hemodynamic_parameter').iloc[0]['TE']
DCM['delays'] = np.zeros(3)
DCM['down_sample_rate_u'] = down_sample_rate_u
DCM['down_sample_rate_y'] = down_sample_rate_y

U = {}
U['dt'] = float(du.get('t_delta') * down_sample_rate_u)
U['name'] = ['input_0', 'input_1']
U['u'] = du.get('u')[::int(down_sample_rate_u), :]
DCM['U'] = U

Y = {}
Y['dt'] = float(du.get('t_delta') * down_sample_rate_y)
Y['y'] = du.get('y')[::int(down_sample_rate_y), :]
Y['y_noised'] = du.get('y_noised')[::int(down_sample_rate_y), :]
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
DCM['du_data'] = du._secured_data

DCM_initial = DCM
scipy.io.savemat(SAVE_PATH_MAT, mdict={'DCM_initial': DCM_initial})
