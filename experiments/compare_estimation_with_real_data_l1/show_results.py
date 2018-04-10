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

def prepare_bar_plot(du_rnn, spm, variable):
    if variable in ['A', 'C']:
        filter = (np.abs(du_rnn.get(variable).flatten()) +
                  np.abs(spm[variable.lower()].flatten())) > 0
        # ticket
        label_indexes = itertools.product(np.linspace(1, du_rnn.get('n_node'), du_rnn.get('n_node'), dtype=int),
                                          np.linspace(1, du_rnn.get('n_node'), du_rnn.get('n_node'), dtype=int))
        ticket = [variable + str(v) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_rnn = du_rnn.get(variable).flatten()[filter]
        height_spm = spm[variable.lower()].flatten()[filter]
    elif variable == 'B':
        b_rnn = np.array(du_rnn.get(variable))
        # b_spm = np.rollaxis(spm[variable.lower()], 2)
        b_spm = spm[variable.lower()]

        filter = (np.abs(b_rnn.flatten()) +
                  np.abs(b_spm.flatten())) > 0
        # tickets
        label_indexes = itertools.product(np.linspace(1, du_rnn.get('n_node'), du_rnn.get('n_node'), dtype=int),
                                          np.linspace(1, du_rnn.get('n_node'), du_rnn.get('n_node'), dtype=int),
                                          np.linspace(1, du_rnn.get('n_stimuli'), du_rnn.get('n_stimuli'), dtype=int))
        ticket = [variable + str(v[0]) + str(v[1:]) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_rnn = b_rnn.flatten()[filter]
        height_spm = b_spm.flatten()[filter]
    else:
        raise ValueError

    return [ticket, {'rnn': height_rnn, 'spm':height_spm}]


def translate_tickets(tickets, node_names, stimulus_names):
    translated_tickets = []
    for ticket in tickets:
        digits = [int(v) - 1 for v in ticket if v.isdigit()]
        if ticket[0] == 'A':
            translated_tickets.append(node_names[digits[-1]] + ' -> ' + node_names[digits[0]])
        elif ticket[0] == 'B':
            translated_tickets.append(stimulus_names[digits[0]] + ' on ' +
            node_names[digits[-1]] + ' -> ' + node_names[digits[1]])
        elif ticket[0] == 'C':
            translated_tickets.append(stimulus_names[digits[1]] + ' -> ' + node_names[digits[1]])
    return translated_tickets


def plot_effective_connectivity(original_data, du_rnn, spm):
    node_names = original_data['node_names']
    stimulus_names = original_data['stimulus_names']
    heights = {'rnn': [],
               'spm': []}
    tickets = []

    for variable in ['A', 'B', 'C']:
        ticket, height = prepare_bar_plot(du_rnn, spm, variable)
        tickets = tickets + ticket
        for k in ['rnn', 'spm']:
            heights[k] = np.concatenate((heights[k], height[k]))
    tickets = translate_tickets(tickets, node_names, stimulus_names)

    width = 0.9
    n_bar = len(tickets)
    left = np.array(range(n_bar)) * 3

    plt.bar(left, heights['rnn'], width, label='DCM-RNN')
    plt.bar(left + width, heights['spm'], width, label='DCM-SPM')
    plt.xticks(left + width / 2, tickets, rotation='vertical')
    plt.grid()
    plt.legend()
    plt.ylabel('values')


EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_real_data_l1')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
RESULT_PATH = os.path.join(EXPERIMENT_PATH, 'results')
IMAGE_PATH = os.path.join(EXPERIMENT_PATH, 'images')

DATA_PATH = os.path.join(PROJECT_DIR, 'dcm_rnn', 'resources', 'SPM_data_attention.pkl')
DCM_RNN_RESULT_PATH = os.path.join(RESULT_PATH, 'estimation_dcm_rnn_extended.pkl')
SPM_RESULT_PATH = os.path.join(RESULT_PATH, 'spm_results.mat')

# load
original_data = pickle.load(open(DATA_PATH, 'rb'))
du_rnn = pickle.load(open(DCM_RNN_RESULT_PATH, 'rb'))
spm = sio.loadmat(SPM_RESULT_PATH)
spm['b'] = np.rollaxis(spm['b'], 2)    # correct matlab-python transfer error
n_node = du_rnn.get('n_node')
n_stimuli = du_rnn.get('n_stimuli')

# correct the order of nodes
# spm['y_predicted'] = spm['y_predicted'][:, [2, 0, 1]]
# spm['y_predicted'] = spm['y_predicted'][:, [1, 0, 2]]

# plot input
# plt.figure(figsize=(6, 3), dpi=300)
x_axis = np.array(range(du_rnn.get('y').shape[0])) * du_rnn.get('t_delta')
for i in range(du_rnn.get('n_stimuli')):
    plt.subplot(du_rnn.get('n_stimuli'), 1, i + 1)
    plt.plot(x_axis, du_rnn.extended_data['u_upsampled'][:, i], linewidth=1.0)
    plt.xlim([0, 1510])
    plt.xlabel('time (second)')
    plt.ylabel(original_data['stimulus_names'][i])
    if i < n_node - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
        # plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_PATH, 'input.png'), bbox_inches='tight')


# plot y
y_true = spm['y_true']
y_rnn = du_rnn.resample(du_rnn.extended_data['y_reproduced'], y_true.shape, order=3)
y_spm = spm['y_predicted']
x_axis = np.array(range(y_rnn.shape[0])) * du_rnn.get('t_delta')

# plt.figure(figsize=(8, 4), dpi=300)
plt.figure()
for i in range(n_node):
    plt.subplot(n_node, 1, i + 1)
    plt.plot(x_axis, y_true[:, i], linewidth=1.0, label='True')
    plt.plot(x_axis, y_rnn[:, i], '--', linewidth=1.2, label='DCM-RNN')
    plt.plot(x_axis, y_spm[:, i], '-.', linewidth=1.2, label='DCM-SPM')
    plt.xlim([0, 1510])
    plt.xlabel('time (second)')
    plt.ylabel(original_data['node_names'][i])
    plt.legend(loc=1)
    # plt.xlim([0, 500])
    if i < n_node - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    # plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_PATH, 'fMRIs.png'), bbox_inches='tight')

print('DCM-RNN y rMSE = ' + str(tb.rmse(y_rnn, y_true)))
print('SPM DCM y rMSE = ' + str(tb.rmse(y_spm, y_true)))

## plot the effective connectivity
plt.figure()
plot_effective_connectivity(original_data, du_rnn, spm)
plt.savefig(os.path.join(IMAGE_PATH, 'ABC.pdf'), format='pdf', bbox_inches='tight')

# calculate rmse
connectivity_rnn = combine_abc(du_rnn.get('A'), du_rnn.get('B'), du_rnn.get('C'))
connectivity_spm = combine_abc(spm['a'], spm['b'], spm['c'])

print('DCM-RNN connectivity l1 = ' + str(sum(abs(connectivity_rnn))))
print('SPM DCM connectivity l1 = ' + str(sum(abs(connectivity_spm))))

'''


# plt hemodynamic kernels
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

plt.plot(t_rnn, k_rnn)
plt.plot(t_spm, k_spm)

'''