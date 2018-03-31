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
import time
import scipy.io



'''
du_hat.update_trainable_variables(grads_and_vars, step_size=0)
du_hat.regenerate_data()
y_hat = du_hat.get('y')
loss_prediction = tb.mse(y_hat, du.get('y'))
'''


def calculate_loss(du_hat, y_true, y_noise, loss_weights, hemodynamic_parameter_mean=None):
    # loss y
    y_e = du_hat.get('y') - y_true
    loss_y = loss_weights['y'] * 0.5 * np.sum([np.sum(y_e[:, r] ** 2 * np.exp(y_noise[r])) for r in range(du_hat.get('n_node'))])

    # loss q
    loss_q = loss_weights['q'] * 0.5 * du_hat.get('n_time_point') * np.sum([-y_noise[r] for r in range(du_hat.get('n_node'))])

    # loss prior x
    loss_a = 0.5 * 64 * np.sum(np.square(du_hat.get('A')))
    loss_b = 0.5 * np.sum([np.square(du_hat.get('B')[r]) for r in range(du_hat.get('n_node'))])
    loss_c = 0.5 * np.sum(np.square(du_hat.get('C')))
    loss_prior_x = loss_weights['prior_x'] * (loss_a + loss_b + loss_c)

    # loss prior h
    para_h = np.array(du_hat.get('hemodynamic_parameter'))
    if hemodynamic_parameter_mean is None:
        hemodynamic_parameter_mean = np.array(
            du_hat.get_expanded_hemodynamic_parameter_prior_distributions(du_hat.get('n_node'))[
                'mean'
            ])
    mean_h = hemodynamic_parameter_mean
    loss_prior_h = loss_weights['prior_h'] * 0.5 * np.sum(np.square(para_h - mean_h)) * 256

    # loss prior hyper
    loss_prior_hyper = loss_weights['prior_hyper'] * 0.5 * np.sum(np.square(y_noise - 6)) * 128

    loss_total = loss_y + loss_q + loss_prior_x + loss_prior_h + loss_prior_hyper

    loss = {'total': loss_total,
            'y': loss_y,
            'q': loss_q,
            'prior_x': loss_prior_x,
            'prior_h': loss_prior_h,
            'prior_hyper': loss_prior_hyper,
            }

    return loss

def solve_for_noise_variance(du_hat, y_true, y_noise, loss_weights):
    y_e = du_hat.get('y') - y_true
    output = np.zeros(du_hat.get('n_node'))
    for r in range(du_hat.get('n_node')):
        a = loss_weights['y'] * 0.5 * np.sum(y_e[:, r] ** 2)
        b = loss_weights['q'] * 0.5 * du_hat.get('n_time_point')
        temp = y_noise[r]
        for s in range(512 * 8):
            l_original = a * np.exp(temp) - b * temp + loss_weights['prior_hyper'] * 0.5 * 128 * (temp - 6) ** 2
            d_temp = a * np.exp(temp) - b + loss_weights['prior_hyper'] * 128 * (temp - 6)
            temp = temp - 1 / 512. * np.sign(d_temp)
            l_updated = a * np.exp(temp) - b * temp + loss_weights['prior_hyper'] * 0.5 * 128 * (temp - 6) ** 2
            if l_updated >= l_original:
                break
        output[r] = temp
    return output

'''
ls = []
x_axis = (np.array(range(1000)) - 500) / 1000
for offset in x_axis:
    temp = y_noise[0] + offset
    ls.append(a * np.exp(temp) - b * y_noise[r] + 0.5 * 128 * (temp - 6) ** 2)
plt.plot(x_axis + y_noise[r], ls)

'''



MAX_EPOCHS = 32 * 3
CHECK_STEPS = 1
N_RECURRENT_STEP = 192
DATA_SHIFT = 1
MAX_BACK_TRACK = 4
MAX_CHANGE = 0.002
BATCH_RANDOM_DROP_RATE = 1.
SNR = []

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
SETTINGS['h1_s0_n1'] = {'if_update_h_parameter': True,
                        'if_extended_support': False,
                        'if_noised_y': True}
if SETTINGS[CONDITION]['if_update_h_parameter']:
    trainable_flags = {'Wxx': True,
                       'Wxxu': True,
                       'Wxu': True,
                       'alpha': False,
                       'E0': False,
                       'k': True,
                       'gamma': False,
                       'tao': True,
                       'epsilon': True,
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

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data', 'du_DCM_RNN.pkl')
SAVE_PATH = os.path.join(EXPERIMENT_PATH, 'results', 'estimation_' + CONDITION + '.pkl')
SAVE_PATH_FREE_ENERGY = os.path.join(EXPERIMENT_PATH, 'results', 'free_energy_rnn_' + CONDITION + '.mat')

timer = {}

# load data
du = tb.load_template(DATA_PATH)
if SETTINGS[CONDITION]['if_noised_y']:
    du._secured_data['y'] = du.get('y_noised')
n_segments = mth.ceil(len(du.get('y')) - N_RECURRENT_STEP / DATA_SHIFT)

# specify initialization values, loss weighting factors, and mask (support of effective connectivity)
x_parameter_initial = {}
x_parameter_initial['A'] = - np.eye(du.get('n_node'))
x_parameter_initial['B'] = [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))]
x_parameter_initial['C'] = np.zeros((du.get('n_node'), du.get('n_stimuli')))
# x_parameter_initial['C'][0, 0] = 1.
# x_parameter_initial['C'][1, 1] = 1.
h_parameter_initial = du.get('hemodynamic_parameter')

loss_weighting = {
        'y': 1.,
        'q': 1.,
        'prior_x': 1.,
        'prior_h': 1.,
        'prior_hyper': 1.,
        'prior_Wxx': 64.,
        'prior_Wxxu': 1.,
        'prior_Wxu': 1.,
    }
mask = du.create_support_mask()
if SETTINGS[CONDITION]['if_extended_support']:
    mask['Wxx'] = np.ones((du.get('n_node'), du.get('n_node')))

x_parameter_initial_in_graph = du.calculate_dcm_rnn_x_matrices(x_parameter_initial['A'],
                                                               x_parameter_initial['B'],
                                                               x_parameter_initial['C'],
                                                               du.get('t_delta'))

du_hat = du.initialize_a_training_unit(x_parameter_initial_in_graph['Wxx'],
                                       x_parameter_initial_in_graph['Wxxu'],
                                       x_parameter_initial_in_graph['Wxu'],
                                       np.array(h_parameter_initial))

loss_correction = N_RECURRENT_STEP / du_hat.get('y').shape[0]


# build tensorflow model
timer['build_model'] = time.time()
print('building model')
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
dr.max_back_track_steps = MAX_BACK_TRACK
dr.max_parameter_change_per_iteration = MAX_CHANGE
dr.trainable_flags = trainable_flags
dr.loss_weighting = loss_weighting
dr.loss_correction = loss_correction
dr.build_main_graph_parallel(neural_parameter_initial=x_parameter_initial,
                             hemodynamic_parameter_initial=h_parameter_initial)

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
data_hat = {
    'x_initial': tb.split(du_hat.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'h_initial': tb.split(du_hat.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
}
for k in data_hat.keys():
    data[k] = data_hat[k][: n_segments]
for k in data.keys():
    data[k] = data[k][: n_segments]
batches = tb.make_batches(data['u'], data_hat['x_initial'], data_hat['h_initial'], data['y'],
                          batch_size=dr.batch_size, if_shuffle=True)

print('start session')
timer['start_session'] = time.time()
dr.max_parameter_change_per_iteration = MAX_CHANGE
# start session
isess = tf.InteractiveSession()
isess.run(tf.global_variables_initializer())
dr.update_variables_in_graph(isess, dr.x_parameter_nodes,
                             [x_parameter_initial_in_graph['Wxx']]
                             + x_parameter_initial_in_graph['Wxxu']
                             + [x_parameter_initial_in_graph['Wxu']])

loss_weights = {
    'y': 1.,
    'q': 1.,
    'prior_x': 16. * 2.,
    'prior_h': 16. * 2.,
    'prior_hyper': 16. * 2.
}

isess.run(tf.assign(dr.loss_y_weight, loss_weights['y']))
isess.run(tf.assign(dr.loss_q_weight, loss_weights['q']))
isess.run(tf.assign(dr.loss_prior_x_weight, loss_weights['prior_x']))
isess.run(tf.assign(dr.loss_prior_h_weight, loss_weights['prior_h']))
isess.run(tf.assign(dr.loss_prior_hyper_weight, loss_weights['prior_hyper']))
dr.max_parameter_change_per_iteration = MAX_CHANGE
print('start inference')
loss_differences = []
step_sizes = []
for epoch in range(MAX_EPOCHS):
    print('')
    print('epoch:    ' + str(epoch))
    timer['epoch_' + str(epoch)] = time.time()
    gradients = []

    # bar = progressbar.ProgressBar(max_value=len(batches))
    for batch in batches:
        grads_and_vars = \
            isess.run(dr.grads_and_vars,
                      feed_dict={
                          dr.u_placeholder: batch['u'],
                          dr.x_state_initial: batch['x_initial'],
                          dr.h_state_initial: batch['h_initial'],
                          dr.y_true: batch['y']
                      })
        # collect gradients and logs
        gradients.append(grads_and_vars)

    # updating with back-tracking
    ## collect statistics before updating
    step_size = 0
    y_noise_index = du_hat.variable_names_in_graph.index('y_noise')
    y_noise = grads_and_vars[y_noise_index][1] - grads_and_vars[y_noise_index][0] * step_size
    loss_original = calculate_loss(du_hat, du.get('y'), y_noise, loss_weights, np.array(du.get('hemodynamic_parameter')))
    # loss_original = loss['total']

    ## sum gradients
    grads_and_vars = gradients[-1]
    for idx, grad_and_var in enumerate(grads_and_vars):
        grads_and_vars[idx] = (sum([gv[idx][0] for gv in gradients]), grads_and_vars[idx][1])

    ## apply mask to gradients
    variable_names = [v.name for v in tf.trainable_variables()]
    for idx in range(len(grads_and_vars)):
        variable_name = variable_names[idx]
        grads_and_vars[idx] = (grads_and_vars[idx][0] * dr.support_masks[variable_name], grads_and_vars[idx][1])

    # adaptive step size, making max value change dr.max_parameter_change_per_iteration
    max_gradient = max([np.max(np.abs(np.array(g[0])).flatten()) for g in grads_and_vars])
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
            dr.decrease_max_parameter_change_per_iteration()
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
        y_noise_index = du_hat.variable_names_in_graph.index('y_noise')
        y_noise = grads_and_vars[y_noise_index][1] - grads_and_vars[y_noise_index][0] * step_size

        # y_hat = du_hat.get('y')
        # loss_updated = tb.mse(y_hat, du.get('y'))

        loss_updated = calculate_loss(du_hat, du.get('y'), y_noise, loss_weights, np.array(du.get('hemodynamic_parameter')))
        # loss_updated = loss['total']

    except:
        loss_updated['total'] = float('inf')

    while loss_updated['total'] > loss_original['total'] or np.isnan(loss_updated['total']):
        count += 1
        if count == dr.max_back_track_steps:
            step_size = 0.
            dr.decrease_max_parameter_change_per_iteration()
        else:
            step_size = step_size / 2
        print('step_size=' + str(step_size))
        try:
            '''
            dr.update_variables_in_graph(isess, dr.trainable_variables_nodes,
                                         [-val[0] * step_size + val[1] for val in grads_and_vars])
            Wxx, Wxxu, Wxu, h_parameter_initial = isess.run([dr.Wxx, dr.Wxxu[0], dr.Wxu, dr.h_parameter_initial])
            du_hat = regenerate_data(du, Wxx, Wxxu, Wxu, h_parameter_initial)
            '''

            du_hat.update_trainable_variables(grads_and_vars, step_size)
            du_hat.regenerate_data()
            y_noise_index = du_hat.variable_names_in_graph.index('y_noise')
            y_noise = grads_and_vars[y_noise_index][1] - grads_and_vars[y_noise_index][0] * step_size

            #y_hat = du_hat.get('y')
            #loss_updated = tb.mse(y_hat, du.get('y'))
            loss_updated = calculate_loss(du_hat, du.get('y'), y_noise, loss_weights, np.array(du.get('hemodynamic_parameter')))
        except:
            pass
        if step_size == 0.0:
            break

    # calculate the optimal noise variance
    y_noise = solve_for_noise_variance(du_hat, du.get('y'), y_noise, loss_weights)
    parameters_updated = [-val[0] * step_size + val[1] for val in grads_and_vars]
    y_noise_index = du_hat.variable_names_in_graph.index('y_noise')
    parameters_updated[y_noise_index] = y_noise

    # update parameters in graph with found step_size
    if step_size > 0:
        dr.update_variables_in_graph(isess, dr.trainable_variables_nodes, parameters_updated)

    loss_differences.append(loss_original['total'] - loss_updated['total'])
    step_sizes.append(step_size)

    # regenerate connector data
    data_hat['x_initial'] = tb.split(du_hat.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0)
    data_hat['h_initial'] = tb.split(du_hat.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1)
    batches = tb.make_batches(data['u'], data_hat['x_initial'],
                              data_hat['h_initial'], data['y'],
                              batch_size=dr.batch_size,
                              if_shuffle=True)
    batches = tb.random_drop(batches, ration=BATCH_RANDOM_DROP_RATE)

    report = pd.DataFrame(index=['original', 'updated', 'reduced', 'reduced %'],
                          columns=['loss_total', 'loss_y', 'loss_q', 'loss_prior_x', 'loss_prior_h', 'loss_prior_hyper'])
    for name_long in report.keys():
        name = name_long[5:]
        report[name_long] = [loss_original[name], loss_updated[name],
                                loss_original[name] - loss_updated[name],
                                (loss_original[name] - loss_updated[name]) / loss_original[name]]

    print('applied step size:    ' + str(step_size))
    print(report)

    print(du_hat.get('Wxx'))
    print(du_hat.get('Wxxu'))
    print(du_hat.get('Wxu'))
    print(du_hat.get('hemodynamic_parameter')[['k', 'tao', 'epsilon']])
    print('y_noise: ' + str(y_noise))
    du_hat.y_true = du.get('y')
    du_hat.timer = timer
    du_hat.y_noise = y_noise
    pickle.dump(du_hat, open(SAVE_PATH, 'wb'))
    # save data for free energy calculation
    dcm_rnn_free_energy = {}
    dcm_rnn_free_energy['A'] = du_hat.get('A')
    dcm_rnn_free_energy['B'] = du_hat.get('B')
    dcm_rnn_free_energy['C'] = du_hat.get('C')
    dcm_rnn_free_energy['y'] = du_hat.get('y')
    dcm_rnn_free_energy['transit'] = np.log(np.array(du_hat.get('hemodynamic_parameter')['tao']) / 2.)
    dcm_rnn_free_energy['decay'] = np.log(np.array(du_hat.get('hemodynamic_parameter')['k']) / 0.64)
    dcm_rnn_free_energy['epsilon'] = np.log(np.array(du_hat.get('hemodynamic_parameter')['epsilon']) / 1.)
    dcm_rnn_free_energy['Ce'] = np.array(np.exp(-y_noise))
    scipy.io.savemat(SAVE_PATH_FREE_ENERGY, mdict={'dcm_rnn_free_energy': dcm_rnn_free_energy})
timer['end'] = time.time()
print('optimization finished.')


du_hat.y_true = du.get('y')
du_hat.timer = timer
du_hat.y_noise = y_noise
pickle.dump(du_hat, open(SAVE_PATH, 'wb'))

# save data for free energy calculation
dcm_rnn_free_energy = {}
dcm_rnn_free_energy['A'] = du_hat.get('A')
dcm_rnn_free_energy['B'] = du_hat.get('B')
dcm_rnn_free_energy['C'] = du_hat.get('C')
dcm_rnn_free_energy['y'] = du_hat.get('y')
dcm_rnn_free_energy['transit'] = np.log(np.array(du_hat.get('hemodynamic_parameter')['tao']) / 2.)
dcm_rnn_free_energy['decay'] = np.log(np.array(du_hat.get('hemodynamic_parameter')['k']) / 0.64)
dcm_rnn_free_energy['epsilon'] = np.log(np.array(du_hat.get('hemodynamic_parameter')['epsilon']) / 1.)
dcm_rnn_free_energy['Ce'] = np.array(np.exp(-du_hat.y_noise))
scipy.io.savemat(SAVE_PATH_FREE_ENERGY, mdict={'dcm_rnn_free_energy': dcm_rnn_free_energy})


for i in range(dr.n_region):
    plt.subplot(dr.n_region, 1, i + 1)
    x_axis = du.get('t_delta') * np.array(range(0, du.get('y').shape[0]))
    plt.plot(x_axis, du.get('y')[:, i])
    plt.plot(x_axis, du_hat.get('y')[:, i], '--')


r=2
y_e = du_hat.get('y') - y_true
a = 0.5 * np.sum(y_e[:, r] ** 2)
b = 0.5 * du_hat.get('n_time_point')
ls = []
for i in range(2000):
    temp = (i - 1000) / 1000
    r = 0
    l = a * np.exp(temp) - b * y_noise[r] + 0.5 * 128 * (temp - 6) ** 2
    ls.append(l)
x_axis = (np.array(range(2000)) - 1000) / 1000
plt.plot(x_axis, np.array(ls))

