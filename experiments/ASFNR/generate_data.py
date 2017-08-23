# add mask to gradient
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
du._secured_data['t_delta'] = 1. / 64.
du._secured_data['t_scan'] = 5 * 60
du._secured_data['n_node'] = 3
du._secured_data['n_stimuli'] = 2
du._secured_data['A'] = np.array([[-0.6, 0., 0],
                                  [0.3, -0.6, 0.3],
                                  [0.5, 0., -0.6]])
du._secured_data['B'] = [np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]]),
                         np.array([[0, 0, 0],
                                   [-0.25, 0, 0],
                                   [0, 0, 0]])]
du._secured_data['C'] = np.array([[0.8, 0, 0], [0, 0, 0]]).reshape(3, 2)
du.u_t_low = 2
du.u_t_high = 8
du.u_interval_t_low = 2
du.u_interval_t_high = 8

# adjust hemodynamic parameter according to the ones used in SPM DCM
# with decay = 0 and transit = 0
# k = sd = H(1)*exp(decay) = 0.64
# gamma = H(2) = 0.32
# tao = tt = H(3)*exp(transit) = 2
# alpha = H(4) = 0.32
# E0 = H(5) = 0.4
# nu0 = theta0 = 40.3
# epsilon as in put P.epsilon
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
du.complete_data_unit(if_show_message=False, if_check_property=False)

for i in range(du.get('n_node') + 1):
    plt.subplot(4, 1, i + 1)
    if i == 0:
        plt.plot(du.get('u'))
    else:
        plt.plot(du.get('y')[:, i - 1])

# save data after finding a good u
# core = du.collect_parameter_core()
# data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'core.pkl')
# pickle.dump(core, open(data_path, 'wb'))


## do down sample and up sample, make the data used for DCM-RNN inference
# load data
data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'core.pkl')
core = tb.load_template(data_path)
du_original = tb.DataUnit()
du_original.load_parameter_core(core)
du_original.recover_data_unit()

# process the original very high temporal resolution signal
# re-sample input u to 16 points per second
# re-sample output y to 1 point per second
du = copy.deepcopy(du_original)
target_resolution = 32
du._secured_data['u'] = du_original._secured_data['u'][:: int(1 / du_original.get('t_delta') / target_resolution), :]
target_resolution = 1
du._secured_data['y'] = du_original._secured_data['y'][:: int(1 / du_original.get('t_delta') / target_resolution), :]

# than up-sample output y to 16 points per second
start_resolution = 1
target_resolution = 16
t_axis_original = np.linspace(0, du.get('t_scan'), int(du.get('t_scan') * start_resolution), endpoint=True)
t_axis_target = np.linspace(0, du.get('t_scan'), int(du.get('t_scan') * target_resolution), endpoint=True)
y_temp = du._secured_data['y']
du._secured_data['y'] = np.zeros((int(du.get('t_scan') * target_resolution), du.get('n_node')))
for i in range(du.get('n_node')):
    interpolate_func = interp1d(t_axis_original, y_temp[:, i], kind='cubic')
    du._secured_data['y'][:, i] = interpolate_func(t_axis_target)
du._secured_data['t_delta'] = 1. / 16.
du._secured_data['u'] = du.get('u')[::2]
du.u = du.get('u')
du.y = du.get('y')

data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'du_DCM_RNN.pkl')
pickle.dump(du, open(data_path, 'wb'))



# create DCM structure for SPM DCM
data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'core.pkl')
core = tb.load_template(data_path)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()

max(du.get('y')[::int(64), :][:, 0])

mask = {
    'Wxx': np.ones((du.get('n_node'), du.get('n_node'))),
    'Wxxu': [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))],
    'Wxu': np.zeros((du.get('n_node'), du.get('n_stimuli')))
}
mask['Wxx'][0, 1] = 0
mask['Wxx'][0, 2] = 0
mask['Wxx'][2, 1] = 0
mask['Wxxu'][1][1, 0] = 1
mask['Wxu'][0, 0] = 1
down_sample_rate_u = 4
down_sample_rate_y = 64
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

