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

def combine_abc(a, b, c):
    output = []
    if isinstance(a, np.ndarray):
        output.append(a.flatten())
    if isinstance(b, list):
        for bb in b:
            if isinstance(bb, np.ndarray):
                output.append(bb.flatten())
            elif isinstance(bb, list):
                output.append(bb[0].flatten())
    elif isinstance(b, np.ndarray):
        output.append(b.flatten())
    if isinstance(c, np.ndarray):
        output.append(c.flatten())
    return np.concatenate(output, axis=0)

def prepare_bar_plot(du, du_rnn_w, du_rnn_wo, spm, variable):
    if variable == 'A':
        filter = (np.abs(du.get(variable).flatten()) +
                  np.abs(du_rnn_w.get(variable).flatten()) +
                  np.abs(du_rnn_wo.get(variable).flatten()) +
                  np.abs(spm[variable.lower()].flatten())) > 0
        # ticket
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int))
        ticket = [variable + str(v) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = du.get(variable).flatten()[filter]
        height_rnn_w = du_rnn_w.get(variable).flatten()[filter]
        height_rnn_wo = du_rnn_wo.get(variable).flatten()[filter]
        height_spm = spm[variable.lower()].flatten()[filter]
    elif variable == 'B':
        b_true = np.array(du.get(variable))
        b_rnn_w = np.array(du_rnn_w.get(variable))
        b_rnn_wo = np.array(du_rnn_wo.get(variable))
        # b_spm = np.rollaxis(spm[variable.lower()], 2)
        b_spm = spm[variable.lower()]
        filter = (np.abs(b_true.flatten()) +
                  np.abs(b_rnn_w.flatten()) +
                  np.abs(b_rnn_wo.flatten()) +
                  np.abs(b_spm.flatten())) > 0
        # tickets
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_stimuli'), du.get('n_stimuli'), dtype=int))
        ticket = [variable + str(v[0]) + str(v[1:]) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = b_true.flatten()[filter]
        height_rnn_w = b_rnn_w.flatten()[filter]
        height_rnn_wo = b_rnn_wo.flatten()[filter]
        height_spm = b_spm.flatten()[filter]
    elif variable == 'C':
        filter = (np.abs(du.get(variable).flatten()) +
                  np.abs(du_rnn_w.get(variable).flatten()) +
                  np.abs(du_rnn_wo.get(variable).flatten()) +
                  np.abs(spm[variable.lower()].flatten())) > 0
        # ticket
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_stimuli'), du.get('n_stimuli'), dtype=int))
        ticket = [variable + str(v) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = du.get(variable).flatten()[filter]
        height_rnn_w = du_rnn_w.get(variable).flatten()[filter]
        height_rnn_wo = du_rnn_wo.get(variable).flatten()[filter]
        height_spm = spm[variable.lower()].flatten()[filter]
    else:
        raise ValueError

    return [ticket, {'true': height_true, 'rnn_w': height_rnn_w, 'rnn_wo': height_rnn_wo, 'spm':height_spm}]

def plot_effective_connectivity(du, du_rnn_w, du_rnn_wo, spm):
    heights = {'true': [],
               'rnn_w': [],
               'rnn_wo': [],
               'spm': []}
    tickets = []

    for variable in ['A', 'B', 'C']:
        ticket, height = prepare_bar_plot(du, du_rnn_w, du_rnn_wo, spm, variable)
        tickets = tickets + ticket
        for k in ['true', 'rnn_w', 'rnn_wo', 'spm']:
            heights[k] = np.concatenate((heights[k], height[k]))

    width = 0.9
    n_bar = len(tickets)
    left = np.array(range(n_bar)) * 4 * 1.1

    plt.bar(left, heights['true'], width, label='True')
    plt.bar(left + width, heights['rnn_w'], width, label='DCM-RNN w/')
    plt.bar(left + width * 2, heights['rnn_wo'], width, label='DCM-RNN w/o')
    plt.bar(left + width * 3, heights['spm'], width, label='SPM w/o')
    plt.xticks(left + width, tickets, rotation='vertical')
    plt.grid()
    plt.legend()
    plt.ylabel('values')


CONDITION = 'h1_s0_n0'
EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'estimation_x_nonlinearity_simulated_data')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
RESULT_PATH = os.path.join(EXPERIMENT_PATH, 'results')
IMAGE_PATH = os.path.join(EXPERIMENT_PATH, 'images')

SIMULATION_X_NONLINEARITY = 'relu'
ESTIMATION_X_NONLINEARITY = 'relu'
CORE_PATH = os.path.join(DATA_PATH, 'core_' + SIMULATION_X_NONLINEARITY + '.pkl')
DCM_RNN_RESULT_PATH = os.path.join(RESULT_PATH, 'estimation_' + CONDITION + '.pkl')
SPM_RESULT_PATH = os.path.join(RESULT_PATH, 'saved_data_' + CONDITION + '_simulation_' +
                         SIMULATION_X_NONLINEARITY + '_estimation_None.mat')
DCM_RNN_WO_X_NONLINEARITY = os.path.join(EXPERIMENT_PATH, 'results', 'estimation_' + CONDITION + '_simulation_' +
                         SIMULATION_X_NONLINEARITY + '_estimation_None.pkl')
DCM_RNN_W_X_NONLINEARITY = os.path.join(EXPERIMENT_PATH, 'results', 'estimation_' + CONDITION + '_simulation_' +
                         SIMULATION_X_NONLINEARITY + '_estimation_' + ESTIMATION_X_NONLINEARITY + '.pkl')

# show estimation curves
core = tb.load_template(CORE_PATH)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()
if CONDITION == 'h1_s1_n1':
    y_true = du.get('y_noised')[::128]
else:
    y_true = du.get('y')[::128]
y_true.shape

du_rnn_wo = pickle.load(open(DCM_RNN_WO_X_NONLINEARITY, 'rb'))
y_rnn_wo = du_rnn_wo.get('y')[::32]
y_rnn_wo.shape

du_rnn_w = pickle.load(open(DCM_RNN_W_X_NONLINEARITY, 'rb'))
y_rnn_w = du_rnn_w.get('y')[::32]
y_rnn_w.shape

spm = sio.loadmat(SPM_RESULT_PATH)
spm['b'] = np.rollaxis(spm['b'], 2)    # correct matlab-python transfer error
y_spm = spm['y_predicted']
y_spm.shape



# show simulations
x_axis = np.array(range(0, du.get('u').shape[0])) * du.get('t_delta')
for s in range(du.get('n_stimuli')):
    plt.subplot(du.get('n_stimuli'), 1, s + 1)
    plt.plot(x_axis, du.get('u')[:, s])
    plt.xlabel('time (second)')
    plt.ylabel('stimulus_ ' + str(s))
    plt.xlim([0, 400])
    if s < du.get('n_stimuli') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
plt.savefig(os.path.join(IMAGE_PATH, 'stimuli.png'), bbox_inches='tight')


# show neural activity w/ and w/o relu
du_wo = copy.deepcopy(du)
du_wo._secured_data['x_nonlinearity_type'] = 'None'
du_wo.regenerate_data()

x_axis = np.array(range(0, du.get('x').shape[0])) * du.get('t_delta')
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, du.get('x')[:, i], label='w/ relu',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, du_wo.get('x')[:, i], '--', label='w/o relu', alpha=1, linewidth=1.0)
    plt.xlim([0, 400])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'neural_activity.png'), bbox_inches='tight')


x_axis = np.array(range(0, len(y_true))) * 2
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn_w[:, i], '--', label='DCM-RNN w/ ', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn_wo[:, i], '--', label='DCM-RNN w/o ', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i], '-.', label='SPM w/o ', alpha=1, linewidth=1.0)
    plt.xlim([0, 440])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_curves_' + CONDITION + '_' + SIMULATION_X_NONLINEARITY + '.png'), bbox_inches='tight')

# plot rRMSE for each region and each method, and overall
errors = np.zeros((du.get('n_node') + 1, 3))
for n in range(du.get('n_node')):
    errors[n, 0] = tb.rmse(y_rnn_w[:, n], y_true[:, n])
    errors[n, 1] = tb.rmse(y_rnn_wo[:, n], y_true[:, n])
    errors[n, 2] = tb.rmse(y_spm[:, n], y_true[:, n])
errors[3, 0] = tb.rmse(y_rnn_w, y_true)
errors[3, 1] = tb.rmse(y_rnn_wo, y_true)
errors[3, 2] = tb.rmse(y_spm, y_true)

left = np.array(range(4)) * 4 * 0.9
width = 0.8
tickets = ['node_0', 'node_1', 'node_2', 'overall']
plt.figure()
plt.bar(left + 0 * width, errors[:, 0], label='DCM-RNN w/')
plt.bar(left + 1 * width, errors[:, 1], label='DCM-RNN w/o')
plt.bar(left + 2 * width, errors[:, 2], label='SPM w/o')
plt.xticks(left + width, tickets)
plt.grid()
plt.legend()
plt.ylabel('fMRI reproduction rRMSE')
plt.savefig(os.path.join(IMAGE_PATH, 'y_rRMSE_' + CONDITION + '_' + SIMULATION_X_NONLINEARITY + '.png'), bbox_inches='tight')




## plot the effective connectivity
plt.figure()
plot_effective_connectivity(du, du_rnn_w, du_rnn_wo, spm)
plt.savefig(os.path.join(IMAGE_PATH, 'ABC_' + CONDITION + '_' + SIMULATION_X_NONLINEARITY + '.png'), bbox_inches='tight')

# calculate rmse
connectivity_true = combine_abc(du.get('A'), du.get('B'), du.get('C'))
connectivity_rnn_w = combine_abc(du_rnn_w.get('A'), du_rnn_w.get('B'), du_rnn_w.get('C'))
connectivity_rnn_wo = combine_abc(du_rnn_wo.get('A'), du_rnn_wo.get('B'), du_rnn_wo.get('C'))
connectivity_spm = combine_abc(spm['a'], spm['b'], spm['c'])
print('DCM-RNN w/ connectivity rMSE = ' + str(tb.rmse(connectivity_rnn_w, connectivity_true)))
print('DCM-RNN w/o connectivity rMSE = ' + str(tb.rmse(connectivity_rnn_wo, connectivity_true)))
print('SPM DCM connectivity rMSE = ' + str(tb.rmse(connectivity_spm, connectivity_true)))

print(sum(abs(connectivity_rnn_w - connectivity_true)))
print(sum(abs(connectivity_rnn_wo - connectivity_true)))
print(sum(abs(connectivity_spm - connectivity_true)))




'''
# plt hemodynamic kernels

du.regenerate_data()
t_true, k_true = du.get_hemodynamic_kernel()

response_length = 32
impulse_length = 0.5
parameter_package = du.collect_parameter_for_h_scan()
x = np.zeros((int(response_length / du.get('t_delta')), du.get('n_node')))
impulse_length = int(impulse_length / du.get('t_delta'))
x[:impulse_length, :] = 1
t_axis = np.array(range(0, x.shape[0])) * du.get('t_delta')
parameter_package['x'] = x
du._secured_data['h'] = du.scan_h_parallel(parameter_package)

parameter_package = du.collect_parameter_for_y_scan()
du._secured_data['y'] = du.scan_y_parallel(parameter_package)

plt.plot(t_axis, du._secured_data['h'][:, 0, :])
plt.plot(t_axis, du._secured_data['y'][:, 0])


du.get('hemodynamic_parameter')

du_rnn_wo._secured_data['hemodynamic_parameter']  = du.get('hemodynamic_parameter')
t_rnn, k_rnn = du_rnn_wo.get_hemodynamic_kernel()
du_rnn_wo.get('hemodynamic_parameter')
plt.plot(t_rnn, k_rnn)

TEMPLATE_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data', 'data', 'du_DCM_RNN.pkl')
du = tb.load_template(TEMPLATE_PATH)
du_spm = copy.deepcopy(du)
h_parameter = du_spm.get('hemodynamic_parameter')
h_parameter['k'] = 0.64 * np.exp(spm['decay'])
h_parameter['tao'] = 2 * np.exp(spm['transit'])
h_parameter['epsilon'] = np.ones(spm['transit'].shape) * np.exp(spm['epsilon'])
du_spm._secured_data['hemodynamic_parameter'] = h_parameter
du_spm.regenerate_data()
t_spm, k_spm = du_spm.get_hemodynamic_kernel()


plt.plot(t_true, k_true)
plt.plot(t_rnn, k_rnn)
plt.plot(t_spm, k_spm)
'''
