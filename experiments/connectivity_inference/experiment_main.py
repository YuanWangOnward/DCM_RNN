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

# global setting
MAX_EPOCHS = 2
CHECK_STEPS = 1
N_SEGMENTS = 512
LEARNING_RATE = 0.0002
N_RECURRENT_STEP = 64
DATA_SHIFT = 4
IF_NODE_MODE = False
IF_IMAGE_LOG = True
IF_DATA_LOG = False
LOG_EXTRA_PREFIX = ''

# load in data
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)

# build model
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.learning_rate = LEARNING_RATE
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
neural_parameter_initial = {}

# results from previous step
Wxx = np.array([[0.9383, 0., 0.0043],
                [0.0451, 0.9497, 0.0139],
                [0.0111, 0.0763, 0.9137]])
Wxxu = np.zeros((3, 3))
Wxxu[2, 2] = -0.02
Wxu = np.array([0.025, 0, 0]).reshape(3, 1)

neural_parameter_initial['A'] = (Wxx - np.identity(dr.n_region)) / dr.t_delta
neural_parameter_initial['B'] = [Wxxu / dr.t_delta]
neural_parameter_initial['C'] = Wxu / dr.t_delta

dr.n_recurrent_step = 8
dr.learning_rate = LEARNING_RATE
dr.loss_weighting = {'prediction': 50., 'sparsity': 1, 'prior': 1., 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
dr.trainable_flags = {'Wxx': True,
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
dr.build_main_graph(neural_parameter_initial=neural_parameter_initial)

# prepare data
data = {'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
        'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}
n_segment = min([len(data[x]) for x in data.keys()])
for k in data.keys():
    data[k] = data[k][: min([n_segment, N_SEGMENTS])]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_total_accumulated_list = []
    loss_prediction_accumulated_list = []
    for epoch in range(MAX_EPOCHS):
        x_state_initial = dr.set_initial_neural_state_as_zeros(dr.n_region).astype(np.float32)
        h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region).astype(np.float32)
        loss_total_accumulated = 0
        loss_prediction_accumulated = 0
        for i in range(len(data['u'])):
            _, loss_total, loss_prediction, x_state_initial, h_state_initial = \
                sess.run([dr.train, dr.loss_total, dr.loss_prediction, dr.x_connector, dr.h_connector],
                         feed_dict={
                             dr.u_placeholder: data['u'][i],
                             dr.x_state_initial: x_state_initial,
                             dr.h_state_initial: h_state_initial,
                             dr.y_true: data['y'][i]
                         })
            print("Index:", '%04d' % i, "loss_prediction=", "{:.9f}".format(loss_prediction))
            Wxx = sess.run(dr.Wxx)
            print(Wxx)
            loss_total_accumulated += loss_total
            loss_prediction_accumulated += loss_prediction
        loss_total_accumulated_list.append(loss_total_accumulated)
        loss_prediction_accumulated_list.append(loss_prediction_accumulated)
        if epoch % CHECK_STEPS == 0:
            print("Epoch:", '%04d' % epoch, "loss_total=", "{:.9f}".format(loss_total_accumulated),
                  "loss_prediction=", "{:.9f}".format(loss_prediction_accumulated))
            Wxx = sess.run(dr.Wxx)
            print(Wxx)
    Wxx = sess.run(dr.Wxx)
    Wxxu = sess.run(dr.Wxxu)
    Wxu = sess.run(dr.Wxu)


print("Optimization Finished!")
print(loss_total_accumulated_list)
print(loss_prediction_accumulated_list)
print(Wxx)
print(Wxxu[0])
print(Wxu)

signal_length = tb.merge(data['u'], N_RECURRENT_STEP, DATA_SHIFT).shape[0]
index_range = list(range(signal_length))

package = {}
package["Wxx"] = Wxx
package["Wxxu"] = Wxxu
package["Wxu"] = Wxu
package['initial_x_state'] = np.array([0, 0, 0])
package['u'] = tb.merge(data['u'], N_RECURRENT_STEP, DATA_SHIFT)
x_hat = du.scan_x(package)

package["Wxx"] = du.get('Wxx')
package["Wxxu"] = du.get('Wxxu')
package["Wxu"] = du.get('Wxu')
package['initial_x_state'] = np.array([0, 0, 0])
package['u'] = tb.merge(data['u'], N_RECURRENT_STEP, DATA_SHIFT)
x_true = du.scan_x(package)

plt.plot(x_true)
plt.plot(x_hat, '--')
print('mse x_hat vs x_hat_reproduced:' + str(tb.mse(x_hat, x_true)))
