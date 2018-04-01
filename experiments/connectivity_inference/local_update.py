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

MAX_EPOCHS = 48
CHECK_STEPS = 1
N_RECURRENT_STEP = 128
DATA_SHIFT = int(N_RECURRENT_STEP / 4)
MAX_BACK_TRACK = 16
MAX_CHANGE = 0.001

# print(os.getcwd())
# data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
# du = tb.load_template(data_path)
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
du.u_t_low = 5
du.u_t_high = 5
du.u_interval_t_low = 10
du.u_interval_t_high = 25

du.complete_data_unit(if_show_message=False, if_check_property=False)

# plt.plot(du.get('u'))
# plt.plot(du.get('y'))
MAX_SEGMENTS = mth.ceil(len(du.get('y')) - N_RECURRENT_STEP / DATA_SHIFT)


# specify initialization values, loss weighting factors, and mask (support of effective connectivity)
x_parameter_initial = {}
x_parameter_initial['A'] = - np.eye(du.get('n_node'))
x_parameter_initial['B'] = [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))]
x_parameter_initial['C'] = np.zeros((du.get('n_node'), du.get('n_stimuli')))
x_parameter_initial['C'][0, 0] = 1

h_parameter_inital = du.get_standard_hemodynamic_parameters(du.get('n_node'))
# h_parameter_initial['x_h_coupling'] = 1.

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

x_parameter_initial_in_graph = du.calculate_dcm_rnn_x_matrices(x_parameter_initial['A'],
                                                               x_parameter_initial['B'],
                                                               x_parameter_initial['C'],
                                                               du.get('t_delta'))
du_hat = du.initialize_a_training_unit(x_parameter_initial_in_graph['Wxx'],
                                       x_parameter_initial_in_graph['Wxxu'],
                                       x_parameter_initial_in_graph['Wxu'],
                                       np.array(h_parameter_inital))

# build tensorflow model
print('building model')
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
dr.max_back_track_steps = MAX_BACK_TRACK
dr.max_parameter_change_per_iteration = MAX_CHANGE
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
dr.loss_weights = loss_weighting
dr.build_main_graph(neural_parameter_initial=x_parameter_initial)

# process after building the main graph
dr.support_masks = dr.setup_support_mask(mask)
dr.x_parameter_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=dr.variable_scope_name_x_parameter)]
dr.trainable_variables_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
dr.trainable_variables_names = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
du_hat.variable_names_in_graph = du_hat.parse_variable_names(dr.trainable_variables_names)

# prepare data for training
data = {
    'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
    'x_initial': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'h_initial': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
    'x_whole': tb.split(du.get('x'), n_segment=dr.n_recurrent_step + dr.shift_u_x, n_step=dr.shift_data, shift=0),
    'h_whole': tb.split(du.get('h'), n_segment=dr.n_recurrent_step + dr.shift_x_y, n_step=dr.shift_data, shift=1),
    'h_predicted': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
    'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}
for k in data.keys():
    data[k] = data[k][: MAX_SEGMENTS]

# start session
isess = tf.InteractiveSession()
isess.run(tf.global_variables_initializer())

dr.update_variables_in_graph(isess, dr.x_parameter_nodes,
                             [x_parameter_initial_in_graph['Wxx']]
                             + x_parameter_initial_in_graph['Wxxu']
                             + [x_parameter_initial_in_graph['Wxu']])

print('start inference')
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
        gradients.append(grads_and_vars)
        y_before_train.append(y_predicted_before_training)
        x_connectors.append(x_connector_current)
        h_connectors.append(h_connector_current)
        loss_totals.append(loss_total)

        # record the loss before applying gradient
        du_hat.update_trainable_variables(grads_and_vars, step_size=0)
        du_hat.regenerate_data(data['u'][i], x_connector, h_connector)
        y_hat_original = du_hat.get('y')
        loss_prediction_original = tb.mse(y_hat_original, data['y'][i])

        # adaptive step size, making max value change dr.max_parameter_change_per_iteration
        max_gradient = max([np.max(np.abs(np.array(g[0])).flatten()) for g in grads_and_vars])
        # STEP_SIZE = 0.001
        STEP_SIZE = dr.max_parameter_change_per_iteration / max_gradient
        print('max gradient:    ' + str(max_gradient))
        print('STEP_SIZE:    ' + str(STEP_SIZE))

        # try to find proper step size
        step_size = STEP_SIZE
        count = 0
        du_hat.update_trainable_variables(grads_and_vars, step_size)

        Wxx = du_hat.get('Wxx')
        stable_flag = du.check_transition_matrix(Wxx, 1.)
        while not stable_flag:
            count += 1
            if count == dr.max_back_track_steps:
                step_size = 0.
                # dr.decrease_max_parameter_change_per_iteration()
            else:
                step_size = step_size / 2
            warnings.warn('not stable')
            # print('step_size=' + str(step_size))
            du_hat.update_trainable_variables(grads_and_vars, step_size)
            Wxx = du_hat.get('Wxx')
            stable_flag = du.check_transition_matrix(Wxx, 1.)

        try:
            du_hat.regenerate_data()
            y_hat = du_hat.get('y')
            loss_prediction = tb.mse(y_hat, data['y'][i])
        except:
            loss_prediction = float('inf')

        while loss_prediction > loss_prediction_original or np.isnan(loss_prediction):
            count += 1
            if count == dr.max_back_track_steps:
                step_size = 0.
                # dr.decrease_max_parameter_change_per_iteration()
            else:
                step_size = step_size / 2
            # print('step_size=' + str(step_size))
            try:
                du_hat.update_trainable_variables(grads_and_vars, step_size)
                du_hat.regenerate_data()
                y_hat = du_hat.get('y')
                loss_prediction = tb.mse(y_hat, data['y'][i])
            except:
                pass
            if step_size == 0.0:
                break

        # update parameters in graph with found step_size
        if step_size > 0:
            dr.update_variables_in_graph(isess, dr.trainable_variables_nodes,
                                         [-val[0] * step_size + val[1] for val in grads_and_vars])

        step_sizes.append(step_size)
        loss_differences.append(loss_prediction_original - loss_prediction)

        x_connector_current = x_connector
        h_connector_current = h_connector

        if step_size > 0:
            print('applied step size:    ' + str(step_size))
            print('loss_prediction_original:    ' + str(loss_prediction_original))
            print('reduced prediction:    ' + str(loss_differences[-1]))
            print('reduced prediction persentage:    ' + str(loss_differences[-1] / loss_prediction_original))
            print(du_hat.get('Wxx'))
            print(du_hat.get('Wxxu'))
            print(du_hat.get('Wxu'))

print('optimization finished.')


du_hat2 = du_hat.initialize_a_training_unit(du_hat.get('Wxx'),
                                        du_hat.get('Wxxu'),
                                        du_hat.get('Wxu'),
                                        np.array(du_hat.get('hemodynamic_parameter')))
du_hat2.regenerate_data(du.get('u'),
                        dr.set_initial_neural_state_as_zeros(dr.n_region),
                        dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region))

for i in range(dr.n_region):
    plt.subplot(dr.n_region, 1, i + 1)
    plt.plot(du.get('y')[:, i])
    plt.plot(du_hat2.get('y')[:, i], '--')


