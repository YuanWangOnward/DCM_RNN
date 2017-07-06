import random
import numpy as np
from sklearn import linear_model
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

def reproduce_x(w_hat, w_true, if_plot=True):
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

    if if_plot:
        plt.plot(x_hat, '--')
        plt.plot(x_true)

    return [x_hat, x_true]

# load in data
print('working directory is ' + os.getcwd())
data_path = os.path.join(PROJECT_DIR, 'experiments', 'calculate_abc_from_x', 'data.pkl')
data_package = tb.load_template(data_path)
data = data_package.data
# data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
# du = tb.load_template(data_path)
du = data['key']
signal_length = data['x_true_merged'].data.shape[0]
index_range = range(0, int(signal_length * 3 / 4))

# check whether du and data match
np.testing.assert_array_almost_equal(du.get('u')[index_range], data['u_merged'][index_range])
np.testing.assert_array_almost_equal(du.get('x')[index_range], data['x_true_merged'][index_range])

# short hand for data
x = data['x_hat_merged'].data[index_range]
u = data['u_merged'][index_range]
xu = np.asarray([np.kron(u[i], x[i]) for i in range(len(index_range))])
np.set_printoptions(precision=4)

"""     --------------------     """
# fit with support and element wise sparse penalty
Y = np.transpose(x[1:])
X = np.transpose(np.concatenate([x[:-1], xu[:-1], u[:-1]], axis=1))

alpha_mask = np.ones((3, 7)) * 0.02
alpha_mask[0, 6] = 0.01
alpha_mask[:, 3:6] = 1
alpha_mask[2, 5] = 0.01
alpha_mask[:, 6] = 1
alpha_mask[0, 6] = 0.01

support = np.ones((3, 7))
support[:, 0:3] = 1
support[2, 5] = 1
support[0, 6] = 1

prior = np.zeros((3, 7))
prior[:, :3] = np.identity(3) * 0.90
prior[0, 6] = 0.1

W = np.transpose(
    tb.ista(np.transpose(X), np.transpose(Y),
            alpha_mask=np.transpose(alpha_mask),
            support=np.transpose(support),
            prior=np.transpose(prior)
            ))

print(W[:, :3])
print(du.get('Wxx'))
print(W[:, 3:6])
print(du.get('Wxxu')[0])
print(W[:, 6].reshape(3, 1))
print(du.get('Wxu'))

w_hat = [W[:, :3], [W[:, 3:6]], W[:, 6].reshape(3,1)]
w_true = [du.get('Wxx'), du.get('Wxxu'), du.get('Wxu')]
x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true, if_plot=True)
plt.plot(data['x_hat_merged'].data[index_range])
plt.plot(x_hat_rep, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))
print('mse x_hat vs x_true:' + str(tb.mse(data['x_true_merged'][index_range], x_hat_rep)))

'''
Y = np.transpose(x[1:])
X = np.transpose(np.concatenate([x[:-1], xu[:-1], u[:-1]], x_axis=1))

# fitting
clf = linear_model.Lasso(alpha=0.000005)
clf.fit(X, Y)

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
print(clf.coef_[:, :3] * w_xx_f)
print(du.get('Wxx'))

w_hat = [clf.coef_[:, :3] * w_xx_f, [Wxxu], Wxu]
w_true = [du.get('Wxx'), du.get('Wxxu'), du.get('Wxu')]
x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true, if_plot=False)
plt.plot(data['x_hat_merged'].data[index_range])
plt.plot(x_hat_rep, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))

print('mse x_hat vs x_true:' + str(tb.mse(data['x_true_merged'][index_range], x_hat_rep)))






# form Y
Y = x[1:] - x[: -1]
xu = np.asarray([np.kron(u[i], x[i]) for i in range(n_time_point)])
X = np.concatenate([x[:-1], xu[:-1], u[:-1]], x_axis=1)

# fitting
clf = linear_model.Lasso(alpha=0.000001, tol=0.000001, max_iter=2000)
clf.fit(X, Y)
print(clf.coef_[:, :3] + np.identity(3))
print(du.get('Wxx'))
print(clf.coef_[:, 3:6])
print(du.get('Wxxu')[0])
print(clf.coef_[:, 6].reshape(3,1))
print(du.get('Wxu'))

w_hat = [clf.coef_[:, :3] + np.identity(3), [clf.coef_[:, 3:6]], clf.coef_[:, 6].reshape(3, 1)]
w_true = [du.get('Wxx'), du.get('Wxxu'), du.get('Wxu')]
x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true, if_plot=False)
plt.plot(data['x_hat_merged'].data[index_range])
plt.plot(x_hat_rep, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))
print('mse x_hat vs x_true:' + str(tb.mse(data['x_true_merged'][index_range], x_hat_rep)))

"""-------------------------------"""
# fit with varying sparsity penalty

Y = x[1:] - x[: -1]
xu = np.asarray([np.kron(u[i], x[i]) for i in range(n_time_point)])
w_xx_f = 1.
w_xxu_f = 100.
w_xu_f = 1.
X = np.concatenate([x[:-1] / w_xx_f, xu[:-1] / w_xxu_f, u[:-1] / w_xu_f], x_axis=1)

# fitting
clf = linear_model.Lasso(alpha=0.00001, tol=0.000001, max_iter=2000)
clf.fit(X, Y)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
print(clf.coef_[:, :3] * w_xx_f + np.identity(3))
print(du.get('Wxx') )
print(clf.coef_[:, 3:6] * w_xxu_f)
print(du.get('Wxxu')[0])
print(clf.coef_[:, 6].reshape(3, 1) * w_xu_f)
print(du.get('Wxu'))

w_hat = [clf.coef_[:, :3] * w_xx_f + np.identity(3),
         [clf.coef_[:, 3:6] * w_xxu_f],
         clf.coef_[:, 6].reshape(3, 1) * w_xu_f ]
w_true = [du.get('Wxx'), du.get('Wxxu'), du.get('Wxu')]
x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true, if_plot=False)
plt.plot(data['x_hat_merged'].data[index_range])
plt.plot(x_hat_rep, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))
print('mse x_hat vs x_true:' + str(tb.mse(data['x_true_merged'][index_range], x_hat_rep)))

'''
'''
"""-------------------------------"""
# fit with sparsity penalty and without Wxxu and Wxu
n_time_point = x.shape[0]
xu_ = np.asarray([np.kron(u[i], x[i]) for i in range(n_time_point)])
w_xx_f = 1
Wxxu = np.zeros((3, 3))

xxu_list = np.linspace(-0.04, 0., 9)
xu_list = np.linspace(0.01, 0.05, 9)
result = {}
result['mse'] = 100
result['xxu'] = 0
result['xu'] = 0

for xxu in xxu_list:
    for xu in xu_list:

        Wxxu[2, 2] = xxu
        Wxu = np.array([xu, 0, 0]).reshape(3, 1)

        Y = x[1:] \
            - np.transpose(np.matmul(Wxxu, np.transpose(xu_[:-1]))) \
            - np.transpose(np.matmul(Wxu, np.transpose(u[:-1])))
        X = np.concatenate([x[:-1]], x_axis=1)

        # fitting
        clf = linear_model.Lasso(alpha=0.000005)
        clf.fit(X, Y)

        # testing
        w_hat = [clf.coef_[:, :3] * w_xx_f, [Wxxu], Wxu]
        w_true = [du.get('Wxx'), du.get('Wxxu'), du.get('Wxu')]
        x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true, if_plot=False)
        mse = tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)
        if mse < result['mse']:
            result['mse'] = mse
            result['xxu'] =xxu
            result['xu'] = xu
print('best xxu: ' + str(result['xxu']))
print('best xu: ' + str(result['xu']))
print('best mse: ' + str(result['mse']))

Wxxu[2, 2] = result['xxu']
Wxu = np.array([result['xu'], 0, 0]).reshape(3, 1)
Y = x[1:] \
    - np.transpose(np.matmul(Wxxu, np.transpose(xu_[:-1]))) \
    - np.transpose(np.matmul(Wxu, np.transpose(u[:-1])))
X = np.concatenate([x[:-1]], x_axis=1)

# fitting
clf = linear_model.Lasso(alpha=0.000005)
clf.fit(X, Y)

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
print(clf.coef_[:, :3] * w_xx_f)
print(du.get('Wxx'))

w_hat = [clf.coef_[:, :3] * w_xx_f, [Wxxu], Wxu]
w_true = [du.get('Wxx'), du.get('Wxxu'), du.get('Wxu')]
x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true, if_plot=False)
plt.plot(data['x_hat_merged'].data[index_range])
plt.plot(x_hat_rep, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))
'''

'''
"""     --------------------     """
# fit with support and element wise sparse penalty
tb.ista()
Wxxu[2, 2] = -0.025
Wxu = np.array([0.25, 0, 0]).reshape(3, 1)
Y = x[1:] \
    - np.transpose(np.matmul(Wxxu, np.transpose(xu_[:-1]))) \
    - np.transpose(np.matmul(Wxu, np.transpose(u[:-1])))
X = np.concatenate([x[:-1]], x_axis=1)

# fitting
clf = linear_model.Lasso(alpha=0.000005)
clf.fit(X, Y)

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
print(clf.coef_[:, :3] * w_xx_f)
print(du.get('Wxx'))

w_hat = [clf.coef_[:, :3] * w_xx_f, [Wxxu], Wxu]
w_true = [du.get('Wxx'), du.get('Wxxu'), du.get('Wxu')]
x_hat_rep, x_true_rep = reproduce_x(w_hat, w_true, if_plot=False)
plt.plot(data['x_hat_merged'].data[index_range])
plt.plot(x_hat_rep, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(data['x_hat_merged'].data[index_range], x_hat_rep)))






print('mse x_hat vs x_true:' + str(tb.mse(data['x_true_merged'][index_range], x_hat_rep)))




""" -------------------"""

# create data
X = np.random.rand(128, 4)
W_true = np.random.rand(4, 2)
W_true[W_true < 0.5] = 0
Y = np.matmul(X, W_true)

# test fitting
clf = linear_model.Lasso(alpha=0.0001)
clf.fit(X, Y)
print(np.transpose(clf.coef_))
print(W_true)
'''
