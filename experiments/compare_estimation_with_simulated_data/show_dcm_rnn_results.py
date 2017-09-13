# replot results with and without updating hemodynamic results
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

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'du_DCM_RNN.pkl')
CORE_PATH = os.path.join(EXPERIMENT_PATH, 'core.pkl')
DCM_RNN_RESULT_PATH_1 = os.path.join(EXPERIMENT_PATH, 'estimation_h1_s1_n0.pkl')
DCM_RNN_RESULT_PATH_2 = os.path.join(EXPERIMENT_PATH, 'estimation_h1_s1_n1.pkl')

# show estimation curves
core = tb.load_template(CORE_PATH)
du = tb.DataUnit()
du.load_parameter_core(core)
du.recover_data_unit()
y_true = du.get('y')[::128]
x_axis_true_impulse, y_true_impulse = du.get_hemodynamic_kernel()
y_true.shape


du_rnn_1 = pickle.load(open(DCM_RNN_RESULT_PATH_1, 'rb'))
x_axis_rnn_1_impulse, y_rnn_1_impulse = du.get_hemodynamic_kernel()
y_rnn_1 = du_rnn_1.get('y')[::32]
y_rnn_1.shape


du_rnn_2 = pickle.load(open(DCM_RNN_RESULT_PATH_2, 'rb'))
x_axis_rnn_2_impulse, y_rnn_2_impulse = du.get_hemodynamic_kernel()
y_rnn_2 = du_rnn_2.get('y')[::32]
y_rnn_2.shape


x_axis = np.array(range(0, len(y_true))) * 2
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis, y_true[:, i], label='True',  alpha=1)
    plt.plot(x_axis, y_rnn_1[:, i], '--', label='fixed h_para', alpha=1)
    plt.plot(x_axis, y_rnn_2[:, i], '-.', label='tunable h_para', alpha=1)
    plt.xlim([0, 420])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend(loc=1)
plt.savefig(os.path.join(EXPERIMENT_PATH, 'images', 'y_curves_h01.png'), bbox_inches='tight')

print('DCM-RNN w/o h_parameter updating rMSE=' + str(tb.rmse(y_rnn_1, y_true)))
print('DCM-RNN w/ h_parameter updating rMSE=' + str(tb.rmse(y_rnn_2, y_true)))


## show hemodynamic kernel
for i in range(du.get('n_node')):
    plt.subplot(du.get('n_node'), 1, i + 1)
    plt.plot(x_axis_true_impulse, y_true_impulse[:, i], label='True',  alpha=1)
    plt.plot(x_axis_rnn_1_impulse, y_rnn_1_impulse[:, i], '--', label='fixed h_para', alpha=1)
    plt.plot(x_axis_rnn_2_impulse, y_rnn_2_impulse[:, i], '-.', label='tunable h_para', alpha=1)
    # plt.xlim([0, 420])
    plt.xlabel('time (second)')
    plt.ylabel('node_ ' + str(i))
    if i < du.get('n_node') - 1:
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend(loc=1)
plt.savefig(os.path.join(EXPERIMENT_PATH, 'images', 'h_kernel_h01.png'), bbox_inches='tight')


