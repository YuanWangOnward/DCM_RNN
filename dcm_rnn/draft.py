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
import progressbar
import math as mth


def regenerate_data(du, Wxx, Wxxu, Wxu, h_parameters):
    du_hat = copy.deepcopy(du)
    du_hat._secured_data['n_node'] = 3
    du_hat._secured_data['n_stimuli'] = 3
    du_hat._secured_data['A'] = (Wxx - np.eye(du_hat.get('n_node'))) / dr.t_delta
    du_hat._secured_data['B'] = [Wxxu / dr.t_delta]
    du_hat._secured_data['C'] = Wxu / dr.t_delta

    du_hat._secured_data['t_delta'] = dr.t_delta

    hemodynamic_parameter_temp = du_hat.get_standard_hemodynamic_parameters(dr.n_region)
    temp = h_parameters
    for c in range(temp.shape[1]):
        hemodynamic_parameter_temp[hemodynamic_parameter_temp.columns[c]] = temp[:, c]
    du_hat._secured_data['hemodynamic_parameter_temp'] = hemodynamic_parameter_temp
    du_hat._secured_data['initial_x_state'] = du_hat.set_initial_neural_state_as_zeros(dr.n_region)
    du_hat._secured_data['initial_h_state'] = du_hat.set_initial_hemodynamic_state_as_inactivated(dr.n_region)
    del du_hat._secured_data['Wxx']
    del du_hat._secured_data['Wxxu']
    del du_hat._secured_data['Wxu']
    del du_hat._secured_data['Whh']
    del du_hat._secured_data['Whx']
    del du_hat._secured_data['bh']
    del du_hat._secured_data['Wo']
    del du_hat._secured_data['bo']

    del du_hat._secured_data['x']
    del du_hat._secured_data['h']
    del du_hat._secured_data['y']

    du_hat.complete_data_unit(start_categorty=1, if_check_property=False, if_show_message=False)
    return du_hat


def apply_and_check(isess, grads_and_vars, step_size, u, x_connector, h_connector, y_true):
    dr.update_variables_in_graph(isess, dr.trainable_variables_nodes,
                                 [-val[0] * step_size + val[1] for val in grads_and_vars])
    loss_prediction, x_connector, h_connector \
        = isess.run([dr.loss_prediction, dr.x_connector, dr.h_connector],
                    feed_dict={
                        dr.u_placeholder: u,
                        dr.x_state_initial: x_connector,
                        dr.h_state_initial: h_connector,
                        dr.y_true: y_true
                    })
    return [loss_prediction, x_connector, h_connector]


def check_transition_matrix(Wxx):
    w, v = np.linalg.eig(Wxx)
    if max(w.real) < 1:
        return True
    else:
        return False


def make_batches(u, x, h, y, batch_size=128, if_shuffle=True):
    batches = []
    temp_u = copy.deepcopy(u)
    temp_x = copy.deepcopy(x)
    temp_h = copy.deepcopy(h)
    temp_y = copy.deepcopy(y)

    if if_shuffle:
        index = list(range(len(temp_y)))
        random.shuffle(index)
        temp_u = [temp_u[i] for i in index]
        temp_x = [temp_x[i] for i in index]
        temp_h = [temp_h[i] for i in index]
        temp_y = [temp_y[i] for i in index]


    for i in range(0, len(temp_u), batch_size):
        batch = make_a_batch(temp_u[i: i + batch_size],
                             temp_x[i: i + batch_size],
                             temp_h[i: i + batch_size],
                             temp_y[i: i + batch_size])
        if len(batch['u']) == batch_size and len(batch['x_initial']) == batch_size and \
                        len(batch['h_initial']) == batch_size and len(batch['y']) == batch_size:
            batches.append(batch)
    return batches

def make_a_batch(u, x, h, y):
    batch = {}
    batch['u'] = np.stack(u, axis=0)
    batch['x_initial'] = np.stack([v[0, :] for v in x], axis=0)
    batch['h_initial'] = np.stack([v[0, :, :] for v in h], axis=0)
    batch['y'] = np.stack(y, axis=0)
    return batch

def random_drop(batches, ration=0.5):
    random.shuffle(batches)
    return batches[: int(len(batches) * ration)]

MAX_EPOCHS = 48
CHECK_STEPS = 1
# MAX_SEGMENTS = 128 * 21
N_RECURRENT_STEP = 160
N_SEGMENTS = mth.ceil(2880 / N_RECURRENT_STEP) * N_RECURRENT_STEP
# STEP_SIZE = 0.002 # for 32
# STEP_SIZE = 0.5
# STEP_SIZE = 0.001 # for 64
# STEP_SIZE = 0.001 # 128
# STEP_SIZE = 0.0005 # for 256
# STEP_SIZE = 5e-5
# DATA_SHIFT = int(N_RECURRENT_STEP / 4)
DATA_SHIFT = 4
LEARNING_RATE = 0.01 / N_RECURRENT_STEP
MAX_BACK_TRACK = 10

print('current working directory is ' + os.getcwd())
# PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)

print('building model')
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.learning_rate = LEARNING_RATE
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
neural_parameter_initial = {'A': du.get('A'), 'B': du.get('B'), 'C': du.get('C')}
dr.loss_weighting = {'prediction': 1., 'sparsity': 0.1, 'prior': 10, 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
dr.trainable_flags = {'Wxx': True,
                      'Wxxu': True,
                      'Wxu': True,
                      'alpha': True,
                      'E0': True,
                      'k': True,
                      'gamma': True,
                      'tao': True,
                      'epsilon': False,
                      'V0': False,
                      'TE': False,
                      'r0': False,
                      'theta0': False,
                      'x_h_coupling': False
                      }

# y_hat are re-generated after each global updating
# gradients of each segments are calculated all according to this y_hat
# namely, Wxx, Wxxu, and Wxu are not updated in each segment
Wxx = np.identity(3) * (1 - 1.2 * du.get('t_delta'))
Wxxu = np.zeros((3, 3))
Wxu = np.array([0., 0., 0.]).reshape(3, 1)
Wxu[0] = 0.0625
h_parameters = dr.get_standard_hemodynamic_parameters(dr.n_region)

du_hat = regenerate_data(du, Wxx, Wxxu, Wxu, np.array(h_parameters))

data_hat = {
    'x_initial': tb.split(du_hat.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'h_initial': tb.split(du_hat.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
}
data = {
    'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
    'x_initial': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'h_initial': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
    'x_whole': tb.split(du.get('x'), n_segment=dr.n_recurrent_step + dr.shift_u_x, n_step=dr.shift_data, shift=0),
    'h_whole': tb.split(du.get('h'), n_segment=dr.n_recurrent_step + dr.shift_x_y, n_step=dr.shift_data, shift=1),
    'h_predicted': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
    'y_true': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
    'y_true_float_corrected': []}
for k in data.keys():
    data[k] = data[k][: N_SEGMENTS]

for i in range(len(data['y_true'])):
    parameter_package = du.collect_parameter_for_y_scan()
    parameter_package['h'] = data['h_predicted'][i].astype(np.float32)
    data['y_true_float_corrected'].append(du.scan_y(parameter_package))

N_TEST_SAMPLE = min(N_SEGMENTS, len(data['y_true_float_corrected']))

# make training batches
batches = make_batches(data['u'], data_hat['x_initial'], data_hat['h_initial'], data['y_true_float_corrected'])
