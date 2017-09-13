# plot results
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
import scipy.io
import scipy.io as sio

def resample(PARA_CORE_PATH):
    # load data
    core = tb.load_template(PARA_CORE_PATH)
    du_original = tb.DataUnit()
    du_original.load_parameter_core(core)
    du_original.recover_data_unit()

    # process the original very high temporal resolution signal
    # re-sample input u to 16 points per second
    # re-sample output y to 1 point per second
    du = copy.deepcopy(du_original)
    target_resolution = 64
    du._secured_data['u'] = du_original._secured_data['u'][:: int(1 / du_original.get('t_delta') / target_resolution),
                            :]
    target_resolution = 1
    du._secured_data['y'] = du_original._secured_data['y'][:: int(1 / du_original.get('t_delta') / target_resolution),
                            :]

    # than up-sample output y to 16 points per second
    start_resolution = 1
    target_resolution = 16
    t_axis_original = np.linspace(0, du.get('t_scan'), int(du.get('t_scan') * start_resolution), endpoint=True)
    t_axis_target = np.linspace(0, du.get('t_scan'), int(du.get('t_scan') * target_resolution), endpoint=True)
    y_temp = du._secured_data['y']
    du._secured_data['y'] = np.zeros((int(du.get('t_scan') * target_resolution), du.get('n_node')))
    for i in range(du.get('n_node')):
        interpolate_func = interp1d(t_axis_original, y_temp[:, i], kind='cubic')
        du._secured_data['y'][:, i] = interpolate_func(t_axis_target)
    du._secured_data['t_delta'] = 1. / 16.
    du._secured_data['u'] = du.get('u')[::2]
    du.u = du.get('u')
    du.y = du.get('y')
    return du

def regenerate(PARA_CORE_PATH):
    core = tb.load_template(PARA_CORE_PATH)
    du = tb.DataUnit()
    du.load_parameter_core(core)
    du.recover_data_unit()
    du._secured_data['t_delta'] = 1 / 16.
    du._secured_data['u'] = du._secured_data['u'][::4, :]
    du.regenerate_data()
    return du

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_simulation_with_SPM')
PARA_CORE_PATH = os.path.join(EXPERIMENT_PATH, 'core.pkl')
MAT_PATH = os.path.join(EXPERIMENT_PATH, 'SPM_simulation.mat')

# load simulation by SPM
spm_data = sio.loadmat(MAT_PATH)
x_axis_spm = np.squeeze(np.array(spm_data['t'])) / 16
x_spm = spm_data['x_predicted']
y_spm = spm_data['y_predicted']

# original
du = tb.DataUnit()
du.load_parameter_core(tb.load_template(PARA_CORE_PATH))
du.recover_data_unit()
x_axis_true = np.array(range(0, du.get('u').shape[0])) * du.get('t_delta')
u = du.get('u')
x_true = du.get('x')
y_true = du.get('y')[::64, :]

# resample
du_resampled = resample(PARA_CORE_PATH)
x_axis_resampled = np.array(range(0, du_resampled.get('u').shape[0])) * du_resampled.get('t_delta')
x_resampled = du_resampled.get('x')
y_resampled = du_resampled.get('y')[::16, :]

# regenerate
du_regenerated = regenerate(PARA_CORE_PATH)
x_axis_regenerated = np.array(range(0, du_regenerated.get('u').shape[0])) * du_regenerated.get('t_delta')
x_regenerated = du_regenerated.get('x')
y_regenerated = du_regenerated.get('y')[::16, :]

# plot input
# plt.figure(figsize=(6, 3), dpi=300)
for i in range(du.get('n_stimuli')):
    plt.subplot(du.get('n_stimuli'), 1, i + 1)
    plt.plot(x_axis_true, u[:, i], linewidth=1.0)
    plt.xlabel('time (second)')
    plt.ylabel('stimulus_' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    # plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENT_PATH, 'input.png'), bbox_inches='tight')

# plot x
# plt.figure(figsize=(8, 4), dpi=300)
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis_true, x_true[:, i], linewidth=1.0, label='True')
    plt.plot(x_axis_spm, x_spm[:, i], '-.', linewidth=1.0, label='SPM')
    plt.plot(x_axis_regenerated, x_regenerated[:, i],  '--', linewidth=1.0, label='regenerated')
    plt.xlabel('time (second)')
    plt.ylabel('node_' + str(i))
    plt.legend(loc=1)
    # plt.xlim([0, 500])
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    # plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENT_PATH, 'neural_activity.png'), bbox_inches='tight')


# plot y
# plt.figure(figsize=(8, 4), dpi=300)
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(y_true[:, i], linewidth=1.0, label='True')
    plt.plot(y_spm[:, i], '-.', linewidth=1.0, label='SPM')
    plt.plot(y_regenerated[:, i],  '--', linewidth=1.0, label='regenerated')
    plt.plot(y_resampled[:, i], '--', linewidth=1.0, label='resampled')
    plt.xlabel('time (second)')
    plt.ylabel('node_' + str(i))
    plt.legend(loc=2)
    # plt.xlim([0, 500])
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    # plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENT_PATH, 'fMRIs.png'), bbox_inches='tight')


tb.rmse(y_spm, y_true)
tb.rmse(y_resampled, y_true)
tb.rmse(y_regenerated, y_true)

for i in range(du.get('n_node')):
    print('For node ' + str(i))
    print('fMRI reproduction SPM:' + str(tb.rmse(y_spm[:, i], y_true[:, i])))
    print('fMRI reproduction regeneration:' + str(tb.rmse(y_regenerated[:, i], y_true[:, i])))
    print('fMRI reproduction resampling:' + str(tb.rmse(y_resampled[:, i], y_true[:, i])))

np.sqrt(tb.mse(y_spm, y_true)/tb.mse(y_true, np.array(0)))
