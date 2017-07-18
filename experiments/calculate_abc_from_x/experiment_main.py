import sys
# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib
    sys.path.append('dcm_rnn')
elif '/share/apps/python3/' in sys.executable:
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

def reproduce_x(w_hat, w_true):
    package = {}
    package["Wxx"] = w_hat[0]
    package["Wxxu"] = w_hat[1]
    package["Wxu"] = w_hat[2]
    package['initial_x_state'] = np.array([0, 0, 0])
    package['u'] = data['u_merged'][index_range]
    x_hat = du.scan_x(package)
    package["Wxx"] = w_true[0]
    package["Wxxu"] = w_true[1]
    package["Wxu"] = w_true[2]
    x_true = du.scan_x(package)

    plt.plot(x_hat, '--')
    plt.plot(x_true)

    return [x_hat, x_true]


# load in SPM_data
print('working directory is ' + os.getcwd())
data_path = os.path.join(PROJECT_DIR, 'experiments', 'calculate_abc_from_x', 'data_ideal.pkl')
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

# confirm estimation
signal_length = data['x_hat_merged'].data.shape[0]
plt.plot(data['x_hat_merged'].data[:signal_length], '--')
plt.plot(du.get('x')[:signal_length])


# confirm estimation segments that are used for calculation abc matrices
index_range = range(0, int(signal_length * 3 / 4))
# index_range = range(0, 450)
plt.plot(data['x_hat_merged'].data[index_range], '--')
plt.plot(du.get('x')[index_range])

# calculate W with sparse prior
# solve_for_effective_connection(x, u, prior=None):
# Y = W*X
# T^T = X^T * W^T
# x = SPM_data['x_hat_merged'].SPM_data[index_range]
x = data['x_true_merged'][index_range]
u = data['u_merged'][index_range]
prior = None
# parameters
n_time_point = x.shape[0]
n_node = x.shape[1]
n_stimuli = u.shape[1]

# form Y
Y = np.transpose(x[1:])

# form X and modify Y
X = [x[:-1]]
Y_prior = np.zeros(Y.shape)
xu = np.asarray([np.kron(u[i], x[i]) for i in range(n_time_point)])
if prior is not None:
    if 'Wxxu' in prior.keys():
        Y_prior += np.matmul(np.concatenate(prior['Wxxu'], axis=1), np.transpose(xu[:-1]))
    else:
        X.append(xu[:-1])
    if 'Wxu' in prior.keys():
        Y_prior += np.matmul(prior['Wxu'], np.transpose(u[:-1]))
    else:
        X.append(u[:-1])
else:
    X.append(xu[:-1])
    X.append(u[:-1])

    Y = Y - Y_prior
    X = np.transpose(np.concatenate(X, axis=1))

from sklearn import linear_model


clf = linear_model.Lasso(alpha=0.00001)
clf.fit(np.transpose(X), np.transpose(Y))



w_true = np.concatenate([du.get('Wxx'), du.get('Wxxu')[0], du.get('Wxu')], axis=1)

tb.rmse(np.transpose(clf.coef_), X)
tb.rmse(np.matmul(w_true, np.transpose(clf.coef_)), Y)


print(np.transpose(clf.coef_)[0: 3])
print(np.transpose(clf.coef_)[4: 7])
print(np.transpose(clf.coef_)[-1])

w_hat = [np.transpose(clf.coef_)[0: 3] + np.identity(3), [np.transpose(clf.coef_)[4: 7]], np.transpose(clf.coef_)[-1].reshape(3,1)]
x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true)
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))
print('mse x_hat vs x_true:' + str(tb.mse(data['x_true_merged'][index_range], x_hat_rep)))



# calculate W without prior
w_hat = tb.solve_for_effective_connection(data['x_hat_merged'].data[index_range], data['u_merged'][index_range])
w_true = tb.solve_for_effective_connection(data['x_true_merged'][index_range], data['u_merged'][index_range])

x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true)
plt.plot(data['x_hat_merged'].data[index_range])
plt.plot(x_hat_rep, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))
print('mse x_hat vs x_true:' + str(tb.mse(data['x_true_merged'][index_range], x_hat_rep)))

print(w_hat[0])
print(w_true[0])

print(w_hat[1][0])
print(w_true[1][0])

print(w_hat[2])
print(w_true[2])


# calculate W with Wxxu
Wxxu = np.zeros((3, 3))
Wxxu[2, 2] = -0.025
prior_hat = {'Wxxu': [Wxxu]}
prior_true = {'Wxxu': [Wxxu]}
w_hat = tb.solve_for_effective_connection(
    data['x_hat_merged'].data[index_range], data['u_merged'][index_range], prior_hat)
w_true = tb.solve_for_effective_connection(
    data['x_true_merged'][index_range], data['u_merged'][index_range], prior_true)

reproduce_x(w_hat, w_true)

print("calculate W with Wxxu")
print(w_hat[0])
print(w_true[0])

print(w_hat[1][0])
print(w_true[1][0])

print(w_hat[2])
print(w_true[2])


# calculate W with Wxxu = 0 and Wxu
Wxxu = np.zeros((3, 3))
Wxxu[2, 2] = -0.025
prior_hat = {'Wxxu': [np.zeros((3, 3))], 'Wxu': np.array([0.025, 0, 0]).reshape(3, 1)}
prior_true = {'Wxxu': [Wxxu], 'Wxu': np.array([0.025, 0, 0]).reshape(3, 1)}
w_hat = tb.solve_for_effective_connection(
    data['x_hat_merged'].data[index_range], data['u_merged'][index_range], prior_hat)
# w_hat = tb.solve_for_effective_connection(
#     SPM_data['x_true_merged'][index_range], SPM_data['u_merged'][index_range], prior_hat)
w_true = tb.solve_for_effective_connection(
    data['x_true_merged'][index_range], data['u_merged'][index_range], prior_true)
reproduce_x(w_hat, w_true, False)

print("calculate W with Wxxu and Wxu")
print(w_hat[0])
print(w_true[0])

print(w_hat[1][0])
print(w_true[1][0])

print(w_hat[2])
print(w_true[2])








