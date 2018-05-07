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
import matplotlib.ticker as mtick


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
    if variable == 'A':
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
    elif variable == 'C':
        filter = (np.abs(du.get(variable).flatten()) +
                  np.abs(du_rnn.get(variable).flatten()) +
                  np.abs(spm[variable.lower()].flatten())) > 0
        # ticket
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_stimuli'), du.get('n_stimuli'), dtype=int))
        ticket = [variable + str(v) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = du.get(variable).flatten()[filter]
        height_rnn = du_rnn.get(variable).flatten()[filter]
        height_spm = spm[variable.lower()].flatten()[filter]
    else:
        raise ValueError

    return [ticket, {'true': height_true, 'rnn': height_rnn, 'spm': height_spm}]


def prepare_bar_plot_confidence(du, confidence_range_rnn, confidence_range_spm, variable):
    if variable == 'A':
        filter = (np.abs(du.get(variable).flatten()) +
                  np.abs(confidence_range_rnn[variable.lower()].flatten()) +
                  np.abs(confidence_range_spm[variable.lower()].flatten())) > 0
        # ticket
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int))
        ticket = [variable + str(v) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = du.get(variable).flatten()[filter]
        height_rnn = confidence_range_rnn[variable.lower()].flatten()[filter]
        height_spm = confidence_range_spm[variable.lower()].flatten()[filter]
    elif variable == 'B':
        b_true = np.array(du.get(variable))
        b_rnn = confidence_range_rnn[variable.lower()]
        b_spm = confidence_range_spm[variable.lower()]
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
    elif variable == 'C':
        filter = (np.abs(du.get(variable).flatten()) +
                  np.abs(confidence_range_rnn[variable.lower()].flatten()) +
                  np.abs(confidence_range_spm[variable.lower()].flatten())) > 0
        # ticket
        label_indexes = itertools.product(np.linspace(1, du.get('n_node'), du.get('n_node'), dtype=int),
                                          np.linspace(1, du.get('n_stimuli'), du.get('n_stimuli'), dtype=int))
        ticket = [variable + str(v) for v in label_indexes]
        ticket = [v for i, v in enumerate(ticket) if filter[i]]

        # height
        height_true = du.get(variable).flatten()[filter]
        height_rnn = confidence_range_rnn[variable.lower()].flatten()[filter]
        height_spm = confidence_range_spm[variable.lower()].flatten()[filter]
    else:
        raise ValueError

    return [ticket, {'true': height_true, 'rnn': height_rnn, 'spm': height_spm}]


def plot_effective_connectivity(du, du_rnn, spm, confidence_range_rnn=None, confidence_range_spm=None):
    heights = {'true': [],
               'rnn': [],
               'spm': []}
    tickets = []

    for variable in ['A', 'B', 'C']:
        ticket, height = prepare_bar_plot(du, du_rnn, spm, variable)
        tickets = tickets + ticket
        for k in ['true', 'rnn', 'spm']:
            heights[k] = np.concatenate((heights[k], height[k]))

    if confidence_range_rnn is not None and confidence_range_spm is not None:
        confidences = {'true': [],
                       'rnn': [],
                       'spm': []}
        for variable in ['A', 'B', 'C']:
            _, confidence = prepare_bar_plot_confidence(du, confidence_range_rnn, confidence_range_spm, variable)
            for k in ['true', 'rnn', 'spm']:
                confidences[k] = np.concatenate((confidences[k], confidence[k]))

    width = 0.9
    n_bar = len(tickets)
    left = np.array(range(n_bar)) * 3

    if confidence_range_rnn is not None and confidence_range_spm is not None:
        plt.bar(left, heights['true'], width, label='True')
        plt.bar(left + width, heights['rnn'], width, label='DCM-RNN',
                yerr=confidences['rnn'], error_kw=dict(ecolor='black', lw=1, capsize=3, capthick=1))
        plt.bar(left + width * 2, heights['spm'], width, label='DCM-SPM',
                yerr=confidences['spm'], error_kw=dict(ecolor='black', lw=1, capsize=3, capthick=1))
        plt.xticks(left + width, tickets, rotation='vertical')
        plt.grid()
        plt.legend()
        plt.ylabel('connectivity values')
    else:
        plt.bar(left, heights['true'], width, label='True')
        plt.bar(left + width, heights['rnn'], width, label='DCM-RNN estimation')
        plt.bar(left + width * 2, heights['spm'], width, label='DCM-SPM estimation')
        plt.xticks(left + width, tickets, rotation='vertical')
        plt.grid()
        plt.legend()
        plt.ylabel('connectivity values')


SETTINGS = {
    'if_update_h_parameter': {'value': 1, 'short_name': 'h'},
    'if_extended_support': {'value': 0, 'short_name': 's'},
    'if_noised_y': {'value': 1, 'short_name': 'n'},
    'snr': {'value': 1, 'short_name': 'snr'},
}

if SETTINGS['if_update_h_parameter']['value']:
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

if not SETTINGS['if_noised_y']['value']:
    SETTINGS['snr']['value'] = float("inf")

keys = sorted(SETTINGS.keys())
SAVE_NAME_EXTENTION = '_'.join([SETTINGS[k]['short_name'] + '_' + str(SETTINGS[k]['value']) for k in keys])

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data_2')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
RESULT_PATH = os.path.join(EXPERIMENT_PATH, 'results')
IMAGE_PATH = os.path.join(EXPERIMENT_PATH, 'images')

CORE_PATH = os.path.join(DATA_PATH, 'core.pkl')
DCM_RNN_RESULT_PATH = os.path.join(RESULT_PATH, 'estimation_' + SAVE_NAME_EXTENTION + '.pkl')
SPM_RESULT_PATH = os.path.join(RESULT_PATH, 'saved_data_' + SAVE_NAME_EXTENTION + '.mat')
DCM_RNN_CONFIDENCE_PATH = os.path.join(RESULT_PATH, 'confidence_range_' + SAVE_NAME_EXTENTION + '_rnn.mat')
SPM_CONFIDENCE_PATH = os.path.join(RESULT_PATH, 'confidence_range_' + SAVE_NAME_EXTENTION + '_spm.mat')

IF_SHOW_HEMODYNAMICS = False
IF_SHOW_CROSS_SNR_RESULTS = False

# recover data
core = tb.load_template(CORE_PATH)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()

du_rnn = pickle.load(open(DCM_RNN_RESULT_PATH, 'rb'))

spm = sio.loadmat(SPM_RESULT_PATH)
spm['b'] = np.rollaxis(spm['b'], 2)  # correct matlab-python transfer error

confidence_range_spm = sio.loadmat(SPM_CONFIDENCE_PATH)
confidence_range_spm['b'] = np.rollaxis(confidence_range_spm['b'], 2)
confidence_range_rnn = sio.loadmat(DCM_RNN_CONFIDENCE_PATH)
confidence_range_rnn['b'] = np.rollaxis(confidence_range_rnn['b'], 2)

# show input
u = du_rnn.get('u')
x_axis = np.array(range(0, len(u))) / 16
plt.figure()
for i in range(du.get('n_stimuli')):
    plt.subplot(du.get('n_stimuli'), 1, i + 1)
    plt.plot(x_axis, u[:, i], alpha=1, linewidth=1.0)
    plt.xlim([0, 490])
    plt.xlabel('time (second)')
    plt.ylabel('stimulus_ ' + str(i))
    if i < du.get('n_stimuli') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
plt.savefig(os.path.join(IMAGE_PATH, 'input_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')

# show simulated curves
y_rnn_simulation = du.get('y')
y_spm_simulation = spm['y_spm_simulation']
x_axis = np.array(range(0, len(y_rnn_simulation))) / 64
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_rnn_simulation[:, i], label='DCM-RNN', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm_simulation[:, i], '--', label='DCM-SPM', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn_simulation[:, i] - y_spm_simulation[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 490])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_simulation_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')
print('y simulation rMSE = ' + str(tb.rmse(y_rnn_simulation, y_spm_simulation)))

# check interpolation of RNN
y_true = du.get('y')[::4]
y_interpolated = du_rnn.y_true
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i], '--', label='DCM-RNN', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 490])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
# plt.savefig(os.path.join(IMAGE_PATH, 'y_interpolation_rnn_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')
print('DCM-RNN y interpolation rMSE = ' + str(tb.rmse(y_interpolated, y_true)))

# show estimation curves (RNN)
y_true = du_rnn.y_true
y_rnn = du_rnn.get('y')
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='Observed', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn[:, i], '--', label='RNN pred.', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 490])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_estimation_rnn_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf',
            bbox_inches='tight')
print('DCM-RNN y estimation rMSE = ' + str(tb.rmse(y_rnn, y_true)))

# check interpolation of SPM
y_true = spm['y_spm_simulation'][::4]
y_interpolated = spm['y_true']
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i], '--', label='DCM-SPM', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 490])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
# plt.savefig(os.path.join(IMAGE_PATH, 'y_interpolation_spm_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')
print('DCM-SPM y interpolation rMSE = ' + str(tb.rmse(y_interpolated, y_true)))

# show estimation curves (SPM)
y_true = spm['y_true']
y_spm = spm['y_predicted']
x_axis = np.array(range(0, len(y_true))) / 16
plt.figure()
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='Observed', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i], '--', label='SPM pred.', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 490])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_estimation_spm_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf',
            bbox_inches='tight')
print('DCM-SPM y estimation rMSE = ' + str(tb.rmse(y_spm, y_true)))

## plot the effective connectivity
# load confidence range

plt.figure()
plot_effective_connectivity(du, du_rnn, spm, confidence_range_rnn, confidence_range_spm)
plt.savefig(os.path.join(IMAGE_PATH, 'ABC_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')

# calculate rmse
connectivity_true = combine_abc(du.get('A'), du.get('B'), du.get('C'))
connectivity_rnn = combine_abc(du_rnn.get('A'), du_rnn.get('B'), du_rnn.get('C'))
connectivity_spm = combine_abc(spm['a'], spm['b'], spm['c'])
print('DCM-RNN connectivity rMSE = ' + str(tb.rmse(connectivity_rnn, connectivity_true)))
print('SPM DCM connectivity rMSE = ' + str(tb.rmse(connectivity_spm, connectivity_true)))

sum(abs(connectivity_rnn - connectivity_true))
sum(abs(connectivity_spm - connectivity_true))


# show speed
print('DCM-RNN run time = ' + str(du_rnn.timer['end']-du_rnn.timer['start_session']))

print('DCM-SPM iteration number = ' + str(spm['n_iteration']))
print('DCM-SPM run time = ' + str(spm['estimation_time']))

# running time
rnn_time = [20880.009542942047, 13850.939879894257, 13728.342636108398, 13674.546900987625]
spm_n_iter = [26, 22, 23, 21]
spm_time = [2057.01, 1693.38, 1747.7, 1650.66]


# plt hemodynamic kernels
if IF_SHOW_HEMODYNAMICS:
    plt.figure()
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

    TEMPLATE_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data', 'data',
                                 'du_DCM_RNN.pkl')
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

if IF_SHOW_CROSS_SNR_RESULTS:
    # show how results change with SNR
    # the results in the lists are manually recorded.
    # The raw error values can be found running this file. They will be printed in the console.
    # The raw free energy values can be found running compare_free_energy.m.
    width = 0.8
    snrs = [5, 3, 1]
    e_reproduction_rnn = [0.0982785623273, 0.153844540607, 0.437795959522]
    e_reproduction_spm = [0.0977933660211, 0.151533392635, 0.435351827891]
    e_connectivity_rnn = [0.0394532698174, 0.0718888114636, 0.193798324935]
    e_connectivity_spm = [0.13801080453, 0.170086871558, 0.297217301838]
    f_rnn = [-8.0638e+04, -1.2276e+05, -2.6435e+05]
    f_spm = [-8.3593e+04, -1.2615e+05, -2.7813e+05]

    plt.figure()
    plt.bar(snrs, e_reproduction_rnn, label='DCM-RNN')
    plt.bar([snr + width for snr in snrs], e_reproduction_spm, label='DCM-SPM')
    plt.xticks([snr + width / 2 for snr in snrs], snrs)
    plt.xlabel('SNR')
    plt.ylabel('Reproduction rRMSE')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, 'reproduction_error_cross_snr.pdf'), format='pdf',
                bbox_inches='tight')

    plt.figure()
    plt.bar(snrs, e_connectivity_rnn, label='DCM-RNN')
    plt.bar([snr + width for snr in snrs], e_connectivity_spm, label='DCM-SPM')
    plt.xticks([snr + width / 2 for snr in snrs], snrs)
    plt.xlabel('SNR')
    plt.ylabel('Connectivity rRMSE')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, 'connectivity_error_cross_snr.pdf'), format='pdf',
                bbox_inches='tight')

    fig = plt.figure()
    plt.bar(snrs, [v / 100000 for v in f_rnn], label='DCM-RNN')
    plt.bar([snr + width for snr in snrs], [v / 100000 for v in f_spm], label='DCM-SPM')
    plt.xticks([snr + width / 2 for snr in snrs], snrs)
    plt.xlabel('SNR')
    plt.ylabel("Free energy value ($x10^5$)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, 'free_energy_cross_snr.pdf'), format='pdf',
                bbox_inches='tight')


import time
du_test = tb.DataUnit()
du_test.load_parameter_core(du_rnn.collect_parameter_core())
start_time = time.time()
du_test.recover_data_unit()
print("--- %s seconds ---" % (time.time() - start_time))

