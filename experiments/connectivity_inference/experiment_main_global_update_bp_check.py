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
import copy

np.set_printoptions(precision=4)

# global setting
MAX_EPOCHS = 1
CHECK_STEPS = 1
N_SEGMENTS = 64
N_RECURRENT_STEP = 1
LEARNING_RATE = 0.01 / N_RECURRENT_STEP
DATA_SHIFT = 1
IF_NODE_MODE = True
IF_IMAGE_LOG = False
IF_DATA_LOG = False
LOG_EXTRA_PREFIX = ''

# load in data
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)
du_hat = copy.deepcopy(du)

# build model
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.learning_rate = LEARNING_RATE
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
neural_parameter_initial = {}

# for debugging
Wxx = du.get('Wxx')
Wxxu = du.get('Wxxu')[0]
Wxu = du.get('Wxu')

support = np.zeros((3, 7))
support[:, 0:3] = 1
support[2, 5] = 1
support[0, 6] = 1

# sess.run(tf.assign(dr.Wxx, Wxx))
# sess.run(tf.assign(dr.Wxxu[0], Wxxu))
# sess.run(tf.assign(dr.Wxu, Wxu))

neural_parameter_initial['A'] = (Wxx - np.identity(dr.n_region)) / dr.t_delta
neural_parameter_initial['B'] = [Wxxu / dr.t_delta]
neural_parameter_initial['C'] = Wxu / dr.t_delta

dr.n_recurrent_step = N_RECURRENT_STEP
dr.learning_rate = LEARNING_RATE
dr.loss_weighting = {'prediction': 50., 'sparsity': 0., 'prior': 0., 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
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
        'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
        'h': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data)}
n_segment = min([len(data[x]) for x in data.keys()])
for k in data.keys():
    data[k] = data[k][: min([n_segment, N_SEGMENTS])]

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
temp = du.get('Wxx')
temp[0, 0] = temp[0, 0] - 0.1
support[:, 0:3] = 1
support[0, 0] = 1
sess.run(tf.assign(dr.Wxx, temp))

loss_total_accumulated_list = []
loss_prediction_accumulated_list = []

MAX_UPDATE = 0.01
STEP_SIZE = 0.05

MAX_EPOCHS = 1
# loss_chage
for epoch in range(MAX_EPOCHS):
    loss_chage = []
    gradents = []
    x_state_initial = dr.set_initial_neural_state_as_zeros(dr.n_region).astype(np.float32)
    h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region).astype(np.float32)
    loss_total_accumulated = 0
    loss_prediction_accumulated = 0
    for i in range(len(data['u'])):
        # in every segment,
        x_state_initial_previous = copy.deepcopy(x_state_initial)
        h_state_initial_previous = copy.deepcopy(h_state_initial)
        # sess.run(dr.Wxu)
        loss_previous = sess.run(dr.loss_total,
                                 feed_dict={
                                     dr.u_placeholder: data['u'][i],
                                     dr.x_state_initial: x_state_initial_previous,
                                     dr.h_state_initial: h_state_initial_previous,
                                     dr.y_true: data['y'][i]
                                 })

        grads_and_vars, loss_total, loss_prediction, x_state_initial, h_state_initial = \
            sess.run([dr.grads_and_vars, dr.loss_total, dr.loss_prediction, dr.x_connector, dr.h_connector],
                     feed_dict={
                         dr.u_placeholder: data['u'][i],
                         dr.x_state_initial: x_state_initial,
                         dr.h_state_initial: h_state_initial,
                         dr.y_true: data['y'][i]
                     })

        # process gradient
        processed_gradient = grads_and_vars
        processed_gradient = (
            (support[:, :3] * np.minimum(processed_gradient[0][0] * STEP_SIZE, MAX_UPDATE), processed_gradient[0][1]),
            (support[:, 3:6] * np.minimum(processed_gradient[1][0] * STEP_SIZE, MAX_UPDATE), processed_gradient[1][1]),
            (support[:, -1].reshape(3, 1) * np.minimum(processed_gradient[2][0] * STEP_SIZE, MAX_UPDATE),
             processed_gradient[2][1]))

        # apply gradient
        sess.run(tf.assign(dr.Wxx, processed_gradient[0][0] + processed_gradient[0][1]))
        sess.run(tf.assign(dr.Wxxu[0], processed_gradient[1][0] + processed_gradient[1][1]))
        sess.run(tf.assign(dr.Wxu, processed_gradient[2][0] + processed_gradient[2][1]))

        # calculate loss again
        # sess.run(dr.Wxu)
        loss_current = sess.run(dr.loss_total,
                                feed_dict={
                                    dr.u_placeholder: data['u'][i],
                                    dr.x_state_initial: x_state_initial_previous,
                                    dr.h_state_initial: h_state_initial_previous,
                                    dr.y_true: data['y'][i]
                                })

        loss_chage.append(loss_current - loss_previous)

        gradents.append(grads_and_vars)
        loss_total_accumulated += loss_total
        loss_prediction_accumulated += loss_prediction
        if i % CHECK_STEPS == 0:
            print('.', end='', flush=True)

    print('.')
    loss_total_accumulated_list.append(loss_total_accumulated)
    loss_prediction_accumulated_list.append(loss_prediction_accumulated)

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

du_hat._secured_data['Wxx'] = Wxx
du_hat._secured_data['Wxxu'] = Wxxu
du_hat._secured_data['Wxu'] = Wxu
parameter_package = du_hat.collect_parameter_for_x_scan()
du_hat._secured_data['x'] = du_hat.scan_x(parameter_package)
parameter_package = du_hat.collect_parameter_for_h_scan()
du_hat._secured_data['h'] = du_hat.scan_h(parameter_package)
parameter_package = du_hat.collect_parameter_for_y_scan()
du_hat._secured_data['y'] = du_hat.scan_y(parameter_package)

plt.plot(du.get('y')[index_range])
plt.plot(du_hat.get('y')[index_range], '--')
plt.plot(tb.merge(data['y'], N_RECURRENT_STEP, DATA_SHIFT)[index_range], '*', alpha=0.7)
print('mse x_hat vs x_true:' + str(tb.mse(du.get('y')[index_range], du_hat.get('y')[index_range])))

# plt.plot([val[1, 2] for val in W['Wxx']])
plt.plot(np.asarray([val[0][0].flatten() for val in gradents]))
plt.plot(loss_prediction_accumulated_list)
