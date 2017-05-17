import sys
sys.path.append('dcm_rnn')
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


# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
elif '/share/apps/python3/' in sys.executable:
    PROJECT_DIR = '/home/yw1225/projects/DCM_RNN'
    print("It seems a remote run on NYU HPC")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
else:
    PROJECT_DIR = '.'
    print("Not sure executing machine. Make sure to set PROJECT_DIR properly.")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)

# load in data
print('working directory is ' + os.getcwd())
data_path = os.path.join(PROJECT_DIR, 'experiments', 'calculate_abc_from_x', 'data.pkl')
data_package = tb.load_template(data_path)
data = data_package.data
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)
signal_length = data['x_true_merged'].data.shape[0]

# find minimal length, being enough to identify ABC matrices
w_true_whole = tb.solve_for_effective_connection(du.get('x'), du.get('u'))
for n in range(64, len(du.get('x')), 64):
    print('Now processing ' + str(n))
    w_true = tb.solve_for_effective_connection(du.get('x')[:n], du.get('u')[:n])
    if tb.rmse(w_true[0], w_true_whole[0]) < 0.05:
        print('Found minimal length: ' + str(n))
        print('w_true[0]:')
        print(w_true[0])
        print('w_true_whole[0]:')
        print(w_true_whole[0])
        print('rmse: ' + str(tb.rmse(w_true[0], w_true_whole[0])))
        break
check_length = n


# calculate W
w_hat = tb.solve_for_effective_connection(data['x_hat_merged'].data, data['u_merged'])
w_true = tb.solve_for_effective_connection(data['x_true_merged'], data['u_merged'])





# data['u'] = du.get('u')[: len(data['x_hat_merged'].data)]

signal_length = data['x_true_merged'].shape[0]
plt.plot(data['x_true_merged'][:, 0])
plt.plot(du.get('x')[:signal_length, 0])


w_true = tb.solve_for_effective_connection(du.get('x')[:128], du.get('u')[:128])
w_true[0]





np.testing.assert_array_almost_equal(data['x_true_merged'][:128], du.get('x')[:128], decimal=4)

plt.imshow(data['x_true_merged'][:128])
plt.show()

plt.plot(data['x_hat_merged'].data[:128])
plt.plot(data['u'][:128] / 4)
plt.plot(du.get('u')[:128] / 3, '--')
plt.plot(du.get('x')[:128], '--')



'''
np.testing.assert_array_almost_equal(W[0], du.get('Wxx'), decimal=5)
for s in range(du.get('n_stimuli')):
    np.testing.assert_array_almost_equal(W[1][s], du.get('Wxxu')[s], decimal=5)
np.testing.assert_array_almost_equal(W[2], du.get('Wxu'), decimal=5)
'''

