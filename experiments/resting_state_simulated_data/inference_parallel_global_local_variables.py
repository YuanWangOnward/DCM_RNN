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


MAX_EPOCHS = 12
CHECK_STEPS = 1
N_RECURRENT_STEP = 8
DATA_SHIFT = 2
MAX_BACK_TRACK = 4
MAX_CHANGE = 0.0015
U_UPDATE_FACTOR = 0.5    # the max update of u is MAX_CHANGE * X_UPDATE_FACTOR
BATCH_RANDOM_DROP_RATE = 1.
BATCH_SIZE = 128


SIMULATION_X_NONLINEARITY = 'None'
ESTIMATION_X_NONLINEARITY = 'None'
IF_RESTING_STATE = True

# tf.reset_default_graph()
CONDITION = 'h1_s0_n0'
SETTINGS = {}
SETTINGS['h0_s0_n0'] = {'if_update_h_parameter': False,
                        'if_extended_support': False,
                        'if_noised_y': False}
SETTINGS['h1_s0_n0'] = {'if_update_h_parameter': True,
                        'if_extended_support': False,
                        'if_noised_y': False}
SETTINGS['h1_s1_n0'] = {'if_update_h_parameter': True,
                        'if_extended_support': True,
                        'if_noised_y': False}
SETTINGS['h1_s1_n1'] = {'if_update_h_parameter': True,
                        'if_extended_support': True,
                        'if_noised_y': True}
if SETTINGS[CONDITION]['if_update_h_parameter']:
    trainable_flags = {'Wxx': True,
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
else:
    trainable_flags = {'Wxx': True,
                       'Wxxu': True,
                       'Wxu': True,
                       'alpha': False,
                       'E0': False,
                       'k': False,
                       'gamma': False,
                       'tao': False,
                       'epsilon': False,
                       'V0': False,
                       'TE': False,
                       'r0': False,
                       'theta0': False,
                       'x_h_coupling': False
                       }

trainable_flags['Wxu'] = False

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'resting_state_simulated_data')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data', 'du_DCM_RNN_' + SIMULATION_X_NONLINEARITY + '.pkl')
SAVE_PATH = os.path.join(EXPERIMENT_PATH, 'results', 'estimation_' + CONDITION + '_simulation_' +
                         SIMULATION_X_NONLINEARITY + '_estimation_' + ESTIMATION_X_NONLINEARITY + '.pkl')

# load data
du = tb.load_template(DATA_PATH)
if SETTINGS[CONDITION]['if_noised_y']:
    du._secured_data['y'] = du.get('y_noised')

# specify initialization values, loss weighting factors, and mask (support of effective connectivity)
x_parameter_initial = {}
x_parameter_initial['A'] = - np.eye(du.get('n_node')) * 0.5
x_parameter_initial['B'] = [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))]
x_parameter_initial['C'] = np.zeros((du.get('n_node'), du.get('n_stimuli')))
x_parameter_initial['C'][0, 0] = 1
x_parameter_initial['C'][1, 1] = 1
x_parameter_initial['C'][2, 2] = 1
h_parameter_initial = du.get('hemodynamic_parameter')

loss_weighting = {'prediction': 1., 'sparsity': 1., 'prior': 0.1, 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1., 'u_smooth': 10.,
                  'u_coefficient_sparsity': 0.000, 'u_energy': 0.}
mask = du.create_support_mask()
if SETTINGS[CONDITION]['if_extended_support']:
    mask['Wxx'] = np.ones((du.get('n_node'), du.get('n_node')))

x_parameter_initial_in_graph = du.calculate_dcm_rnn_x_matrices(x_parameter_initial['A'],
                                                               x_parameter_initial['B'],
                                                               x_parameter_initial['C'],
                                                               du.get('t_delta'))
du_hat = du.setup_training_assistant({'A': x_parameter_initial['A'],
                                      'B': x_parameter_initial['B'],
                                      'C': x_parameter_initial['C'],
                                      'hemodynamic_parameter': h_parameter_initial,
                                      'x_nonlinearity_type': ESTIMATION_X_NONLINEARITY,
                                      'u': tb.ArrayWrapper(np.zeros(du.get('u').shape),
                                                           N_RECURRENT_STEP, DATA_SHIFT)})

# build tensorflow model
print('building model')
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
dr.max_back_track_steps = MAX_BACK_TRACK
dr.max_parameter_change_per_iteration = MAX_CHANGE
dr.trainable_flags = trainable_flags
dr.loss_weights = loss_weighting
dr.batch_size = BATCH_SIZE
dr.x_nonlinearity_type = ESTIMATION_X_NONLINEARITY
dr.if_resting_state = IF_RESTING_STATE
dr.build_main_graph(neural_parameter_initial=x_parameter_initial,
                    hemodynamic_parameter_initial=h_parameter_initial)

# process after building the main graph
dr.support_masks = dr.setup_support_mask(mask)
dr.x_parameter_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=dr.variable_scope_name_x_parameter)]
dr.trainable_variables_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
dr.trainable_variables_names = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
du_hat.variable_names_in_graph = du_hat.parse_variable_names(dr.trainable_variables_names)
du_hat.global_variable_indexes = copy.deepcopy(du_hat.parse_variable_names(dr.trainable_variables_names))
# du_hat.global_variable_indexes.remove('u_entire')
du_hat.global_variable_indexes = [du_hat.variable_names_in_graph.index(n) for n in du_hat.global_variable_indexes]
# du_hat.local_variable_indexes = [0]

# prepare data for training
data = {
    'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}
data_hat = {
    'u_index': tb.split_index(du_hat.get('u').shape, n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
    'u': tb.split(du_hat.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'x_initial': tb.split(du_hat.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'h_initial': tb.split(du_hat.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
}
batches = tb.make_batches(data_hat['u'], data_hat['x_initial'], data_hat['h_initial'], data['y'],
                          batch_size=dr.batch_size, extra=['index'], if_shuffle=False)


print('start session')
# start session
isess = tf.InteractiveSession()
isess.run(tf.global_variables_initializer())
'''
dr.update_variables_in_graph(isess, dr.x_parameter_nodes,
                             [x_parameter_initial_in_graph['Wxx']]
                             + x_parameter_initial_in_graph['Wxxu']
                             + [x_parameter_initial_in_graph['Wxu']])
'''
dr.update_variables_in_graph(isess, dr.x_parameter_nodes,
                             [x_parameter_initial_in_graph['Wxx']]
                             + x_parameter_initial_in_graph['Wxxu']
                             )
isess.run(tf.assign(dr.loss_weighting_in_graph['u_smooth'], 100.))
isess.run(tf.assign(dr.loss_weighting_in_graph['u_energy'], 1.))
isess.run(tf.assign(dr.loss_weighting_in_graph['prior'], 0.02))
isess.run(tf.assign(dr.loss_weighting_in_graph['sparsity'], 0.0))

print('start inference')
loss_differences = []
step_sizes = []
epoch = 0
for epoch in range(MAX_EPOCHS):
    print('')
    print('epoch:    ' + str(epoch))
    gradients = []

    # bar = progressbar.ProgressBar(max_value=len(batches))
    for batch in batches:
        grads_and_vars = \
            isess.run(dr.grads_and_vars,
                      feed_dict={
                          dr.u_index_place_holder: batch['index'],
                          dr.x_state_initial: batch['x_initial'],
                          dr.h_state_initial: batch['h_initial'],
                          dr.y_true: batch['y']
                      })
        # collect gradients and logs
        gradients.append(grads_and_vars)
    ## sum gradients
    grads_and_vars = du_hat.sum_gradients(gradients, variable_names_in_graph=None)
    ## apply mask to gradients
    variable_names = [v.name for v in tf.trainable_variables()]
    for idx in range(len(grads_and_vars)):
        variable_name = variable_names[idx]
        grads_and_vars[idx] = (grads_and_vars[idx][0] * dr.support_masks[variable_name], grads_and_vars[idx][1])

    ## collect statistics before updating
    # du_hat.update_trainable_variables(grads_and_vars, step_size=0)
    # du_hat.regenerate_data()
    y_hat_original = du_hat.get('y')
    loss_prediction_original = tb.mse(y_hat_original, du.get('y'))
    loss_prediction_original_backup = loss_prediction_original
    # print('loss_prediction_original = ' + str(loss_prediction_original))

    # adaptive step size, making max value change dr.max_parameter_change_per_iteration
    max_gradient = max([np.max(np.abs(np.array(g[0])).flatten()) for g in
                        [grads_and_vars[i] for i in du_hat.local_variable_indexes]])
    STEP_SIZE = dr.max_parameter_change_per_iteration / max_gradient * U_UPDATE_FACTOR
    print('max gradient of local variables:    ' + str(max_gradient))
    print('Initial step size:    ' + str(STEP_SIZE))

    # try to find proper step size
    step_size = STEP_SIZE
    count = 0
    du_hat.update_trainable_variables(grads_and_vars, step_size, update_parameters=['u_entire'])
    try:
        du_hat.regenerate_data()
        y_hat = du_hat.get('y')
        loss_prediction = tb.mse(y_hat, du.get('y'))
    except:
        loss_prediction = float('inf')
    while loss_prediction > loss_prediction_original or np.isnan(loss_prediction):
        count += 1
        if count == dr.max_back_track_steps:
            step_size = 0.
            # dr.decrease_max_parameter_change_per_iteration()
        else:
            step_size = step_size / 2
        print('step_size=' + str(step_size))
        try:
            du_hat.update_trainable_variables(grads_and_vars, step_size)
            du_hat.regenerate_data()
            y_hat = du_hat.get('y')
            loss_prediction = tb.mse(y_hat, du.get('y'))
        except:
            pass
        if step_size == 0.0:
            break

    if step_size > 0:
        dr.update_variables_in_graph(isess,
                                     [dr.trainable_variables_nodes[i] for i in du_hat.local_variable_indexes],
                                     [-val[0] * step_size + val[1] for val in
                                      [grads_and_vars[i] for i in du_hat.local_variable_indexes]])

    if epoch > 0:
        # adaptive step size, making max value change dr.max_parameter_change_per_iteration
        max_gradient = max([np.max(np.abs(np.array(g[0])).flatten()) for g in
                            [grads_and_vars[i] for i in du_hat.global_variable_indexes]])
        STEP_SIZE = dr.max_parameter_change_per_iteration / max_gradient
        print('max gradient of global variables:    ' + str(max_gradient))
        print('Initial step size:    ' + str(STEP_SIZE))
        # try to find proper step size
        step_size = STEP_SIZE
        count = 0
        loss_prediction_original = loss_prediction
        du_hat.update_trainable_variables(grads_and_vars, step_size,
                                          update_parameters=[du_hat.variable_names_in_graph[i] for i in
                                                             du_hat.global_variable_indexes])
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
            print('step_size=' + str(step_size))
            du_hat.update_trainable_variables(grads_and_vars, step_size)
            Wxx = du_hat.get('Wxx')
            stable_flag = du.check_transition_matrix(Wxx, 1.)

        try:
            # du_hat = regenerate_data(du, Wxx, Wxxu, Wxu, h_parameter_initial)
            du_hat.regenerate_data()
            y_hat = du_hat.get('y')
            loss_prediction = tb.mse(y_hat, du.get('y'))
        except:
            loss_prediction = float('inf')

        while loss_prediction > loss_prediction_original or np.isnan(loss_prediction):
            count += 1
            if count == dr.max_back_track_steps:
                step_size = 0.
                # dr.decrease_max_parameter_change_per_iteration()
            else:
                step_size = step_size / 2
            print('step_size=' + str(step_size))
            try:
                du_hat.update_trainable_variables(grads_and_vars, step_size,
                                                  update_parameters=[du_hat.variable_names_in_graph[i] for i in
                                                                     du_hat.global_variable_indexes])
                du_hat.regenerate_data()
                y_hat = du_hat.get('y')
                loss_prediction = tb.mse(y_hat, du.get('y'))
            except:
                pass
            if step_size == 0.0:
                break

        # update parameters in graph with found step_size
        if step_size > 0:
            dr.update_variables_in_graph(isess,
                                         [dr.trainable_variables_nodes[i] for i in du_hat.global_variable_indexes],
                                         [-val[0] * step_size + val[1] for val in
                                          [grads_and_vars[i] for i in du_hat.global_variable_indexes]])


    loss_differences.append(loss_prediction_original_backup - loss_prediction)

    # regenerate connector data
    data_hat['x_initial'] = tb.split(du_hat.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0)
    data_hat['h_initial'] = tb.split(du_hat.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1)
    batches = tb.make_batches(data_hat['u'], data_hat['x_initial'], data_hat['h_initial'], data['y'],
                              batch_size=dr.batch_size, extra=['index'], if_shuffle=False)
    batches = tb.random_drop(batches, ration=BATCH_RANDOM_DROP_RATE)

    print('applied step size:    ' + str(step_size))
    print('loss_prediction_original:    ' + str(loss_prediction_original_backup))
    print('reduced prediction:    ' + str(loss_differences[-1]))
    print('reduced prediction persentage:    ' + str(loss_differences[-1] / loss_prediction_original_backup))
    print(du_hat.get('Wxx'))
    # print(du_hat.get('Wxxu'))
    # print(du_hat.get('Wxu'))
    print(du_hat.get('hemodynamic_parameter'))
    pickle.dump(du_hat, open(SAVE_PATH, 'wb'))

print('optimization finished.')


for i in range(dr.n_region):
    plt.subplot(dr.n_region, 1, i + 1)
    x_axis = du.get('t_delta') * np.array(range(0, du.get('y').shape[0]))
    plt.plot(x_axis, du.get('y')[:, i])
    plt.plot(x_axis, du_hat.get('y')[:, i], '--')


grads_and_vars, loss_total, loss_u_energy, loss_u_smooth, loss_sparsity, loss_prediction= \
            isess.run([dr.grads_and_vars,
                       dr.loss_total,
                       dr.loss_u_energy,
                       dr.loss_u_smooth,
                       dr.loss_sparsity,
                       dr.loss_prediction],
                      feed_dict={
                          dr.u_index_place_holder: batch['index'],
                          dr.x_state_initial: batch['x_initial'],
                          dr.h_state_initial: batch['h_initial'],
                          dr.y_true: batch['y']
                      })



'''
grads_and_vars, x_monitor, x_monitor_stacked, y_predicted_stacked = \
            isess.run([dr.grads_and_vars, dr.x_monitor, dr.x_monitor_stacked, dr.y_predicted_stacked],
                      feed_dict={
                          dr.u_placeholder: batch['u'],
                          dr.x_state_initial: batch['x_initial'],
                          dr.h_state_initial: batch['h_initial'],
                          dr.y_true: batch['y']
                      })

i = 60
print(batch['u'][i][:4,:])
print(batch['x_initial'][i])
print(x_monitor_stacked[i, :4,:])
print('')
print(np.matmul(du_hat.get('Wxx'),  batch['x_initial'][i]) + np.matmul(du_hat.get('Wxu'), batch['u'][i][0,:]))
print(x_monitor_stacked[i, 1,:])

'''

