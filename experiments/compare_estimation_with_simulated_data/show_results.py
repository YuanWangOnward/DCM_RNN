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


SETTINGS = {
'if_update_h_parameter': {'value': 1, 'short_name': 'h'},
'if_extended_support': {'value': 0, 'short_name': 's'},
'if_noised_y': {'value': 1, 'short_name': 'n'},
'snr': {'value': 3, 'short_name': 'snr'},
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


EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
RESULT_PATH = os.path.join(EXPERIMENT_PATH, 'results')
IMAGE_PATH = os.path.join(EXPERIMENT_PATH, 'images')

CORE_PATH = os.path.join(DATA_PATH, 'core.pkl')
DCM_RNN_RESULT_PATH = os.path.join(RESULT_PATH, 'estimation_' + SAVE_NAME_EXTENTION + '.pkl')
SPM_RESULT_PATH = os.path.join(RESULT_PATH, 'saved_data_' + SAVE_NAME_EXTENTION + '.mat')

IF_SHOW_HEMODYNAMICS = False
IF_SHOW_CROSS_SNR_RESULTS =False

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
    plt.xlim([0, 410])
    plt.xlabel('time (second)')
    plt.ylabel('stimulus_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
plt.savefig(os.path.join(IMAGE_PATH, 'input_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')


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
    plt.xlim([0, 410])
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
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i], '--', label='DCM-RNN', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_interpolated[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 410])
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
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn[:, i], '--', label='DCM-RNN', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_rnn[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 410])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_estimation_rnn_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')
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
    plt.xlim([0, 410])
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
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i], '--', label='DCM-SPM', alpha=1, linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i] - y_true[:, i], '-.', label='Error', alpha=1, linewidth=1.0)
    plt.xlim([0, 410])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
plt.savefig(os.path.join(IMAGE_PATH, 'y_estimation_spm_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')
print('DCM-SPM y estimation rMSE = ' + str(tb.rmse(y_spm, y_true)))


## plot the effective connectivity
plt.figure()
plot_effective_connectivity(du, du_rnn, spm)
plt.savefig(os.path.join(IMAGE_PATH, 'ABC_' + SAVE_NAME_EXTENTION + '.pdf'), format='pdf', bbox_inches='tight')

du_temp = copy.deepcopy(du_rnn)

def calculate_loss(du_hat, y_true, y_noise, loss_weights, hemodynamic_parameter_mean=None):
    # loss y
    y_e = du_hat.get('y') - y_true
    loss_y = loss_weights['y'] * 0.5 * np.sum([np.sum(y_e[:, r] ** 2 * np.exp(y_noise[r])) for r in range(du_hat.get('n_node'))])

    # loss q
    loss_q = loss_weights['q'] * 0.5 * du_hat.get('n_time_point') * np.sum([y_noise[r] for r in range(du_hat.get('n_node'))])

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
loss_weights = {
    'y': 1.,
    'q': 1.,
    'prior_x': 16. * 2.,
    'prior_h': 16. * 2.,
    'prior_hyper': 16. * 2.
}



def calculate_loss(du_hat, y_true, noise_lambda, loss_weights=None, prior_mean=None, prior_variance=None):

    if loss_weights is None:
        loss_weights = {
            'y': 1.,
            'q': 1.,
            'prior_x': 1.,
            'prior_h': 1.,
            'prior_hyper': 1.
        }

    if prior_mean is None:
        prior_mean = {
            'A': np.zeros(du_hat.get('A').shape),
            'B': [np.zeros(du_hat.get('A').shape) for _ in range(du_hat.get('n_node'))],
            'C': np.zeros((du_hat.get('n_node'), du_hat.get('n_stimuli'))),
            'hemodynamic_parameter': du_hat.get_standard_hemodynamic_parameters(du_hat.get('n_node')),
            'noise_lambda': np.ones(du_hat.get('n_node')) * 6    # variance = (exp(lambda))^(-1)
        }

    if prior_variance is None:
        prior_variance = {
            'A': np.ones(du_hat.get('A').shape) / 64.,
            'B': [np.ones(du_hat.get('A').shape) for _ in range(du_hat.get('n_node'))],
            'C': np.ones((du_hat.get('n_node'), du_hat.get('n_stimuli'))),
            'hemodynamic_parameter':
                np.square(du_hat.get_expanded_hemodynamic_parameter_prior_distributions(du_hat.get('n_node'))['std']),
            'noise_lambda': np.ones(du_hat.get('n_node')) / 128.  # variance = (exp(lambda))^(-1)
        }
        prior_variance['hemodynamic_parameter'][prior_variance['hemodynamic_parameter'] == 0.] = 1/256

    # loss y (reproduction error)
    y_e = du_hat.get('y') - y_true
    loss_y = loss_weights['y'] * 0.5 * np.sum(
        [np.sum(y_e[:, r] ** 2 * np.exp(noise_lambda[r])) for r in range(du_hat.get('n_node'))])

    # loss q (log|\Sigma|)
    loss_q = loss_weights['q'] * 0.5 * du_hat.get('n_time_point') \
                * np.sum([np.exp(noise_lambda[r]) for r in range(du_hat.get('n_node'))])

    # loss prior x
    e_a = du_hat.get('A') - prior_mean['A']
    e_b = [du_hat.get('B')[r] - prior_mean['B'][r] for r in range(du_hat.get('n_node'))]
    e_c = du_hat.get('C') - prior_mean['C']
    loss_a = 0.5 * np.sum(np.square(e_a) / prior_variance['A'])
    loss_b = 0.5 * np.sum([np.square(e_b[r]) / prior_variance['B'][r] for r in range(du_hat.get('n_node'))])
    loss_c = 0.5 * np.sum(np.square(e_c) / prior_variance['C'])
    loss_prior_x = loss_weights['prior_x'] * (loss_a + loss_b + loss_c)

    # loss prior h
    e_h = np.array(du_hat.get('hemodynamic_parameter')) - prior_mean['hemodynamic_parameter']
    loss_prior_h = loss_weights['prior_h'] * 0.5 * np.sum(np.sum(np.square(e_h) / prior_variance['hemodynamic_parameter']))


    # loss prior hyper
    e_hyper = noise_lambda - prior_mean['noise_lambda']
    loss_prior_hyper = loss_weights['prior_hyper'] * 0.5 * np.sum(np.square(e_hyper) / prior_variance['noise_lambda'])

    # total loss
    loss_total = loss_y + loss_q + loss_prior_x + loss_prior_h + loss_prior_hyper

    loss = {'total': loss_total,
            'y': loss_y,
            'q': loss_q,
            'prior_x': loss_prior_x,
            'prior_h': loss_prior_h,
            'prior_hyper': loss_prior_hyper,
            }

    return loss


# calculate rmse
connectivity_true = combine_abc(du.get('A'), du.get('B'), du.get('C'))
connectivity_rnn = combine_abc(du_rnn.get('A'), du_rnn.get('B'), du_rnn.get('C'))
connectivity_spm = combine_abc(spm['a'], spm['b'], spm['c'])
print('DCM-RNN connectivity rMSE = ' + str(tb.rmse(connectivity_rnn, connectivity_true)))
print('SPM DCM connectivity rMSE = ' + str(tb.rmse(connectivity_spm, connectivity_true)))

sum(abs(connectivity_rnn - connectivity_true))
sum(abs(connectivity_spm - connectivity_true))


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


if IF_SHOW_CROSS_SNR_RESULTS:
    # show how results change with SNR
    # the results in the lists are manually recorded.
    # The raw error values can be found running this file. They will be printed in the console.
    # The raw free energy values can be found running compare_free_energy.m.
    width = 0.8
    snrs = [5, 3, 1]
    e_reproduction_rnn = [0.0981675973274, 0.154117626012, 0.437791342386]
    e_reproduction_spm = [0.0977933660211, 0.151533392635, 0.435351827891]
    e_connectivity_rnn = [0.0306194736943, 0.0770575092325, 0.187405521345]
    e_connectivity_spm = [0.13801080453, 0.170086871558, 0.297217301838]
    f_rnn = [-8.3593e+04, -1.2615e+05, -2.7813e+05]
    f_spm = [-8.0643e+04, -1.2282e+05, -2.6437e+05]

    plt.figure()
    plt.bar(snrs, e_reproduction_rnn, label='DCM-RNN')
    plt.bar([snr + width for snr in snrs], e_reproduction_spm, label='DCM-SPM')
    plt.xticks([snr + width/2 for snr in snrs], snrs)
    plt.xlabel('SNR')
    plt.ylabel('Reproduction rRMSE')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, 'reproduction_error_cross_snr.pdf'), format='pdf',
                bbox_inches='tight')

    plt.figure()
    plt.bar(snrs, e_connectivity_rnn, label='DCM-RNN')
    plt.bar([snr + width for snr in snrs], e_connectivity_spm, label='DCM-SPM')
    plt.xticks([snr + width/2 for snr in snrs], snrs)
    plt.xlabel('SNR')
    plt.ylabel('Connectivity rRMSE')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, 'connectivity_error_cross_snr.pdf'), format='pdf',
                bbox_inches='tight')

    fig = plt.figure()
    plt.bar(snrs, [v / 100000 for v in f_rnn], label='DCM-RNN')
    plt.bar([snr + width for snr in snrs], [v/100000 for v in f_spm], label='DCM-SPM')
    plt.xticks([snr + width / 2 for snr in snrs], snrs)
    plt.xlabel('SNR')
    plt.ylabel("Free energy value ($x10^5$)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, 'free_energy_cross_snr.pdf'), format='pdf',
                bbox_inches='tight')
