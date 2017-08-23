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
from scipy.interpolate import interp1d
import scipy.io as sio

# check forward pass
data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'core.pkl')
core = tb.load_template(data_path)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()
y_true = du.get('y')[::64]

data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'du_DCM_RNN.pkl')
du_rnn = pickle.load(open(data_path, 'rb'))
y_resampled = du_rnn.get('y')[::16]
du_rnn.regenerate_data()
y_regenerated = du_rnn.get('y')[::16]

data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'smp_simulation.mat')
y_spm = sio.loadmat(data_path)['y']



x_axis = np.array(range(0, len(du.get('u')))) * du.get('t_delta')
for i in range(du.get('n_stimuli')):
    plt.subplot(du.get('n_node') + du.get('n_stimuli'), 1, i + 1)
    plt.plot(x_axis, du.get('u')[:, i], linewidth=1.0)
    plt.ylabel('input_' + str(i))
    plt.xlim([0, 430])
    plt.gca().axes.get_xaxis().set_visible(False)
x_axis = np.array(range(0, len(y_true)))
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node') + du.get('n_stimuli'), 1, du.get('n_stimuli') + i + 1)
    plt.plot(x_axis, y_true[:, i], label='True', linewidth=1.0)
    plt.plot(x_axis, y_resampled[:, i], '--', label='Re-sampled', linewidth=1.0)
    # plt.plot(x_axis, y_regenerated[:, i], '--', label='Re-generated', linewidth=1.0)
    plt.plot(x_axis, y_spm[:, i], '-.', label='SPM', linewidth=1.0)
    plt.xlim([0, 430])
    plt.xlabel('time (second)')
    plt.ylabel('fMRI_node_' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend(prop={'size': 10})
# plt.tight_layout()



# show estimation curves
data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'core.pkl')
core = tb.load_template(data_path)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()
y_true = du.get('y')[::64]
y_true.shape

data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'estimation_parallel.pkl')
du_rnn = pickle.load(open(data_path, 'rb'))
y_rnn = du_rnn.get('y')[::16]
y_rnn.shape

data_path = os.path.join(PROJECT_DIR, 'experiments', 'ASFNR', 'smp_predicted.mat')
y_spm = sio.loadmat(data_path)['y_predicted']
y_spm.shape

x_axis = np.array(range(0, len(y_true)))
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1)
    plt.plot(x_axis, y_rnn[:, i], '--', label='DCM-RNN', alpha=1)
    plt.plot(x_axis, y_spm[:, i], '-.', label='DCM', alpha=1)
    plt.xlim([0, 420])
    plt.xlabel('time (second)')
    plt.ylabel('fMRI_node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()

np.mse(1,2)

mse = ((y_true.flatten() - y_spm.flatten()) ** 2).mean()

(du_rnn.get('Wxx') - np.eye(3)) * 16

(du_rnn.get('Wxxu')[1]) * 16

(du_rnn.get('Wxu')) * 16

