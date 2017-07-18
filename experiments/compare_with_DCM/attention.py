# add mask to gradient
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

DATA_DIR = os.path.join(PROJECT_DIR, 'dcm_rnn', 'resources', 'SPM_data_attention.pkl')

MAX_EPOCHS = 16
CHECK_STEPS = 1
N_SEGMENTS = 600
N_RECURRENT_STEP = 512
STEP_SIZE = 0.001  # 128
DATA_SHIFT = int(N_RECURRENT_STEP / 4)
TARGET_TEMPORAL_RESOLUTION = 1. / 16

# load SPM attention SPM_data from resources
SPM_data = pickle.load(open(DATA_DIR, 'rb'))

# process SPM_data, up sample u and y to 16 frame/second
t_total = SPM_data['TR'] * SPM_data['y'].shape[0]
t_axis_original = np.linspace(0, t_total, t_total / SPM_data['TR'], endpoint=True)
t_axis_new = np.linspace(0, t_total, t_total / TARGET_TEMPORAL_RESOLUTION, endpoint=True)
SPM_data['y_upsampled'] = np.zeros((len(t_axis_new), SPM_data['y'].shape[1]))
for i in range(SPM_data['y'].shape[1]):
    interpolate_func = interp1d(t_axis_original, SPM_data['y'][:, i], kind='cubic')
    SPM_data['y_upsampled'][:, i] = interpolate_func(t_axis_new)
SPM_data['u_upsampled'] = np.zeros((len(t_axis_new), SPM_data['u'].shape[1]))
for i in range(SPM_data['u'].shape[1]):
    interpolate_func = interp1d(t_axis_original, SPM_data['u'][:, i])
    SPM_data['u_upsampled'][:, i] = interpolate_func(t_axis_new)
#plt.subplot(2, 1, 1)
#plt.plot(SPM_data['y_upsampled'])
#plt.subplot(2, 1, 2)
#plt.plot(SPM_data['u_upsampled'])

# build dcm_rnn model
importlib.reload(tfm)
tf.reset_default_graph()
dr = tfm.DcmRnn()
dr.n_region = 3
dr.t_delta = 1. / 16
dr.n_stimuli = 3
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
dr.loss_weighting = {'prediction': 1., 'sparsity': 0.1, 'prior': 0.1, 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
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
A = np.eye(dr.n_region) * -0.5
B = [np.zeros((dr.n_region, dr.n_region))] * dr.n_region
C = np.zeros((dr.n_region, dr.n_stimuli))
C[0, 0] = 1
neural_parameter_initial = {'A': A, 'B': B, 'C': C}
hemodynamic_parameter_initial = dr.get_standard_hemodynamic_parameters(dr.n_region).astype(np.float32)
hemodynamic_parameter_initial['x_h_coupling'] = 1.
dr.build_main_graph(neural_parameter_initial=neural_parameter_initial,
                    hemodynamic_parameter_initial=hemodynamic_parameter_initial)

# process after building the main graph
mask = {dr.Wxx.name: np.ones((dr.n_region, dr.n_region)),
        dr.Wxxu[0].name: np.zeros((dr.n_region, dr.n_region)),
        dr.Wxxu[1].name: np.zeros((dr.n_region, dr.n_region)),
        dr.Wxxu[2].name: np.zeros((dr.n_region, dr.n_region)),
        dr.Wxu.name: np.zeros((dr.n_region, dr.n_stimuli)).reshape(dr.n_region, dr.n_stimuli)}
mask[dr.Wxx.name][0, 2] = 0
mask[dr.Wxx.name][2, 0] = 0
mask[dr.Wxxu[1].name][1, 0] = 1
mask[dr.Wxxu[2].name][0, 1] = 1
mask[dr.Wxu.name][0, 0] = 1
dr.support_masks = dr.setup_support_mask(mask)
dr.x_parameter_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=dr.variable_scope_name_x_parameter)]
dr.trainable_variables_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

# prepare SPM_data for training
data = {
    'u': tb.split(SPM_data['u_upsampled'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
    'y': tb.split(SPM_data['y_upsampled'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
}
for k in data.keys():
    data[k] = data[k][: N_SEGMENTS]
N_TEST_SAMPLE = min(N_SEGMENTS, len(data['y']))


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

isess = tf.InteractiveSession()
isess.run(tf.global_variables_initializer())

isess.run(tf.assign(dr.Wxx, np.zeros((dr.n_region, dr.n_region))))

loss_differences = []
loss_totals = []
y_before_train = []
y_after_train = []
x_connectors = []
h_connectors = []
gradients = []
step_sizes = []
for epoch in range(MAX_EPOCHS):
    x_connector_current = dr.set_initial_neural_state_as_zeros(dr.n_region)
    h_connector_current = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region)
    for i in range(len(data['y'])):
        print('current processing ' + str(i))

        grads_and_vars, x_connector, h_connector, loss_prediction, loss_total, y_predicted_before_training = \
            isess.run([dr.grads_and_vars, dr.x_connector, dr.h_connector,
                       dr.loss_prediction, dr.loss_total, dr.y_predicted],
                      feed_dict={
                          dr.u_placeholder: data['u'][i],
                          dr.x_state_initial: x_connector_current,
                          dr.h_state_initial: h_connector_current,
                          dr.y_true: data['y'][i]
                      })

        # apply mask to gradients
        variable_names = [v.name for v in tf.trainable_variables()]
        for idx in range(len(grads_and_vars)):
            variable_name = variable_names[idx]
            grads_and_vars[idx] = (grads_and_vars[idx][0] * dr.support_masks[variable_name], grads_and_vars[idx][1])

        # logs
        loss_prediction_original = loss_prediction
        gradients.append(grads_and_vars)
        y_before_train.append(y_predicted_before_training)
        x_connectors.append(x_connector_current)
        h_connectors.append(h_connector_current)
        loss_totals.append(loss_total)

        # updating with back-tracking
        step_size = STEP_SIZE
        loss_prediction, x_connector, h_connector = \
            apply_and_check(isess, grads_and_vars, step_size, data['u'][i],
                            x_connector_current,
                            h_connector_current,
                            data['y'][i])

        count = 0
        Wxx = isess.run(dr.Wxx)
        stable_flag = check_transition_matrix(Wxx)
        while not stable_flag:
            count += 1
            if count == 20:
                step_size = 0
            else:
                step_size = step_size / 2
            warnings.warn('not stable')
            print('step_size=' + str(step_size))
            Wxx = -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]
            stable_flag = check_transition_matrix(Wxx)

        while (loss_prediction > loss_prediction_original or np.isnan(loss_prediction)):
            count += 1
            if count == 20:
                step_size = 0.
            else:
                step_size = step_size / 2
            print('step_size=' + str(step_size))
            loss_prediction, x_connector, h_connector = \
                apply_and_check(isess, grads_and_vars,
                                step_size, data['u'][i], x_connector_current,
                                h_connector_current, data['y'][i])
            if step_size == 0.0:
                break

        dr.update_variables_in_graph(isess, dr.trainable_variables_nodes,
                                     [-val[0] * step_size + val[1] for val in grads_and_vars])

        y_predicted_after_training = isess.run(dr.y_predicted,
                                               feed_dict={
                                                   dr.u_placeholder: data['u'][i],
                                                   dr.x_state_initial: x_connector_current,
                                                   dr.h_state_initial: h_connector_current,
                                                   dr.y_true: data['y'][i]
                                               })

        y_after_train.append(y_predicted_after_training)
        step_sizes.append(step_size)
        x_connector_current = x_connector
        h_connector_current = h_connector
        loss_differences.append(loss_prediction_original - loss_prediction)
        Wxx, Wxxu, Wxu = isess.run([dr.Wxx, dr.Wxxu, dr.Wxu])

        print(loss_totals[-1])
        print(loss_differences[-1])
        print(Wxx)
        print(Wxxu)
        # print(Wxx + Wxxu)
        print(Wxu)

print('optimization finished.')





