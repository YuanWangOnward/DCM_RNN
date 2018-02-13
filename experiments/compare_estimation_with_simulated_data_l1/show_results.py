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


def prepare_bar_plot(du, du_rnn, spm, variable):
    if variable in ['A', 'C']:
        filter = (np.abs(du.get(variable).flatten()) +
                  np.abs(du_rnn.get(variable).flatten()) +
                  np.abs(spm[variable.lower()].flatten())) > 0
        # ticket
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int))
        ticket = [variable + str(v) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = du.get(variable).flatten()[filter]
        height_rnn = du_rnn.get(variable).flatten()[filter]
        height_spm = spm[variable.lower()].flatten()[filter]
    elif variable == 'B':
        b_true = np.array(du.get(variable))
        b_rnn = np.array(du_rnn.get(variable))
        # b_spm = np.rollaxis(spm[variable.lower()], 2)
        b_spm = spm[variable.lower()]
        filter = (np.abs(b_true.flatten()) +
                  np.abs(b_rnn.flatten()) +
                  np.abs(b_spm.flatten())) > 0
        # tickets
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_stimuli'), du.get('n_stimuli'), dtype=int))
        ticket = [variable + str(v[0]) + str(v[1:]) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = b_true.flatten()[filter]
        height_rnn = b_rnn.flatten()[filter]
        height_spm = b_spm.flatten()[filter]
    else:
        raise ValueError

    return [ticket, {'true': height_true, 'rnn': height_rnn, 'spm':height_spm}]

def plot_effective_connectivity(du, du_rnn, spm):
    heights = {'true': [],
               'rnn': [],
               'spm': []}
    tickets = []

    for variable in ['A', 'B', 'C']:
        ticket, height = prepare_bar_plot(du, du_rnn, spm, variable)
        tickets = tickets + ticket
        for k in ['true', 'rnn', 'spm']:
            heights[k] = np.concatenate((heights[k], height[k]))

    width = 0.9
    n_bar = len(tickets)
    left = np.array(range(n_bar)) * 3

    plt.bar(left, heights['true'], width, label='True')
    plt.bar(left + width, heights['rnn'], width, label='DCM-RNN estimation')
    plt.bar(left + width * 2, heights['spm'], width, label='DCM-SPM estimation')
    plt.xticks(left + width, tickets, rotation='vertical')
    plt.grid()
    plt.legend()
    plt.ylabel('values')

CONDITION = 'h1_s2_n1'
EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data_l1')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
RESULT_PATH = os.path.join(EXPERIMENT_PATH, 'results')
IMAGE_PATH = os.path.join(EXPERIMENT_PATH, 'images')

CORE_PATH = os.path.join(DATA_PATH, 'core.pkl')
DCM_RNN_RESULT_PATH = os.path.join(RESULT_PATH, 'estimation_' + CONDITION + '.pkl')
SPM_RESULT_PATH = os.path.join(RESULT_PATH, 'saved_data_' + CONDITION + '.mat')


# recover data
core = tb.load_template(CORE_PATH)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()

du_rnn = pickle.load(open(DCM_RNN_RESULT_PATH, 'rb'))

spm = sio.loadmat(SPM_RESULT_PATH)
spm['b'] = np.rollaxis(spm['b'], 2)    # correct matlab-python transfer error


# show input
u = du_rnn.get('u')
x_axis = np.array(range(0, len(u))) / 16
plt.figure()
for i in range(du.get('n_stimuli')):
    plt.subplot(du.get('n_stimuli'), 1, i + 1)
    plt.plot(x_axis, u[:, i], alpha=1, linewidth=1.0)
    plt.xlim([0, 660])
    plt.xlabel('time (second)')
    plt.ylabel('stimulus_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
plt.savefig(os.path.join(IMAGE_PATH, 'input_' + CONDITION + '.png'), bbox_inches='tight')


# show simulated curves
y_rnn_simulation = du.get('y')
y_spm_simulation = spm['y_spm_simulation']
x_axis = np.array(range(0, len(y_rnn_simulation))) / 64
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_rnn_simulation[:, i], label='DCM-RNN',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm_simulation[:, i], '--', label='DCM-SPM', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn_simulation[:, i] - y_spm_simulation[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 660])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_simulation_' + CONDITION + '.png'), bbox_inches='tight')
print('y simulation rMSE = ' + str(tb.rmse(y_rnn_simulation, y_spm_simulation)))


# check interpolation of RNN
y_true = du.get('y')[::4]
y_interpolated = du_rnn.y_true
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i], '--', label='DCM-RNN', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 660])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
# plt.savefig(os.path.join(IMAGE_PATH, 'y_interpolation_rnn_' + CONDITION + '.png'), bbox_inches='tight')
print('DCM-RNN y interpolation rMSE = ' + str(tb.rmse(y_interpolated, y_true)))


# show estimation curves (RNN)
y_true = du_rnn.y_true
y_rnn = du_rnn.get('y')
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn[:, i], '--', label='DCM-RNN', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 660])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_estimation_rnn_' + CONDITION + '.png'), bbox_inches='tight')
print('DCM-RNN y estimation rMSE = ' + str(tb.rmse(y_rnn, y_true)))



# check interpolation of SPM
y_true = spm['y_spm_simulation'][::4]
y_interpolated = spm['y_true']
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i], '--', label='DCM-SPM', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 660])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
# plt.savefig(os.path.join(IMAGE_PATH, 'y_interpolation_spm_' + CONDITION + '.png'), bbox_inches='tight')
print('DCM-SPM y interpolation rMSE = ' + str(tb.rmse(y_interpolated, y_true)))



# show estimation curves (SPM)
y_true = spm['y_true']
y_spm = spm['y_predicted']
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i], '--', label='DCM-SPM', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 660])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_estimation_spm_' + CONDITION + '.png'), bbox_inches='tight')
print('DCM-SPM y estimation rMSE = ' + str(tb.rmse(y_spm, y_true)))


## plot the effective connectivity
plt.figure()
plot_effective_connectivity(du, du_rnn, spm)
plt.savefig(os.path.join(IMAGE_PATH, 'ABC_' + CONDITION + '.png'), bbox_inches='tight')

# calculate rmse
connectivity_true = combine_abc(du.get('A'), du.get('B'), du.get('C'))
connectivity_rnn = combine_abc(du_rnn.get('A'), du_rnn.get('B'), du_rnn.get('C'))
connectivity_spm = combine_abc(spm['a'], spm['b'], spm['c'])
print('DCM-RNN connectivity rMSE = ' + str(tb.rmse(connectivity_rnn, connectivity_true)))
print('SPM DCM connectivity rMSE = ' + str(tb.rmse(connectivity_spm, connectivity_true)))

sum(abs(connectivity_rnn - connectivity_true))
sum(abs(connectivity_spm - connectivity_true))

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

du_rnn._secured_data['hemodynamic_parameter'] = du.get('hemodynamic_parameter')
t_rnn, k_rnn = du_rnn.get_hemodynamic_kernel()
du_rnn.get('hemodynamic_parameter')
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

