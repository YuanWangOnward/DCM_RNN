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

MAX_EPOCHS = 1
CHECK_STEPS = 1
N_SEGMENTS = 32
N_RECURRENT_STEP = 64
LEARNING_RATE = 0.01 / N_RECURRENT_STEP
DATA_SHIFT = 4
N_TEST_SAMPLE_MAX = 32

print(os.getcwd())
PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)

dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.learning_rate = LEARNING_RATE
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
neural_parameter_initial = {'A': du.get('A'), 'B': du.get('B'), 'C': du.get('C')}
dr.loss_weighting = {'prediction': 1., 'sparsity': 0., 'prior': 0., 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
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

data = {
    'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
    'x_initial': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'h_initial': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
    'x_whole': tb.split(du.get('x'), n_segment=dr.n_recurrent_step + dr.shift_u_x, n_step=dr.shift_data, shift=0),
    'h_whole': tb.split(du.get('h'), n_segment=dr.n_recurrent_step + dr.shift_x_y, n_step=dr.shift_data, shift=1),
    'h_predicted': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
    'y_true': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
    'y_true_float_corrected': []}

for k in data.keys():
    data[k] = data[k][: N_SEGMENTS]

for i in range(len(data['y_true'])):
    parameter_package = du.collect_parameter_for_y_scan()
    parameter_package['h'] = data['h_predicted'][i].astype(np.float32)
    data['y_true_float_corrected'].append(du.scan_y(parameter_package))

N_TEST_SAMPLE = min(N_TEST_SAMPLE_MAX, len(data['y_true_float_corrected']))

isess = tf.InteractiveSession()


def apply_and_check(isess, grads_and_vars, step_size, u, x_connector, h_connector, y_true):
    isess.run([tf.assign(dr.Wxx, -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]),
               tf.assign(dr.Wxxu[0], -grads_and_vars[1][0] * step_size + grads_and_vars[1][1]),
               tf.assign(dr.Wxu, -grads_and_vars[2][0] * step_size + grads_and_vars[2][1])])
    loss_prediction, x_connector, h_connector \
        = isess.run([dr.loss_prediction, dr.x_connector, dr.h_connector],
                    feed_dict={
                        dr.u_placeholder: u,
                        dr.x_state_initial: x_connector,
                        dr.h_state_initial: h_connector,
                        dr.y_true: y_true
                    })
    return [loss_prediction, x_connector, h_connector]


def check_transition_matrix(Wxx):
    w, v = np.linalg.eig(Wxx)
    if max(w.real) < 1:
        return True
    else:
        return False


isess.run(tf.global_variables_initializer())
STEP_SIZE = 0.001
Wxx = du.get('Wxx')

Wxx[0, 0] = Wxx[0, 0] * 0.9
isess.run(tf.assign(dr.Wxx, Wxx))

gradient_sum = 0
checking_loss = []
y_before_train = []
y_after_train = []
x_connectors = []
h_connectors = []
for epoch in range(MAX_EPOCHS):
    x_connector_current = dr.set_initial_neural_state_as_zeros(dr.n_region)
    h_connector_current = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region)
    for i in range(len(data['y_true_float_corrected'])):
        print('current processing ' + str(i))
        # print('u:')
        # print(data['u'][i])
        # print('x_initial:')
        # print(x_connector_current)
        # print('h_initial:')
        # print(h_connector_current)
        # print('y_true:')
        # print(data['y_true_float_corrected'][i])


        grads_and_vars, x_connector, h_connector, loss_prediction, y_predicted_before_training = \
            isess.run([dr.grads_and_vars, dr.x_connector, dr.h_connector, dr.loss_prediction, dr.y_predicted],
                      feed_dict={
                          dr.u_placeholder: data['u'][i],
                          dr.x_state_initial: x_connector_current,
                          dr.h_state_initial: h_connector_current,
                          dr.y_true: data['y_true_float_corrected'][i]
                      })
        loss_prediction_original = loss_prediction
        gradient_sum += grads_and_vars[0][0]
        y_before_train.append(y_predicted_before_training)
        x_connectors.append(x_connector_current)
        h_connectors.append(h_connector_current)

        # print('r=' + str(r) + ' c=' + str(c) + ' i=' + str(i) +
        #       ' loss_prediction=' + str(loss_prediction))
        # for item in grads_and_vars:
        #    print(item)


        # updating with back-tracking
        step_size = STEP_SIZE
        loss_prediction, x_connector, h_connector = \
            apply_and_check(isess, grads_and_vars, step_size, data['u'][i],
                            x_connector_current,
                            h_connector_current,
                            data['y_true_float_corrected'][i])

        count = 0
        while (loss_prediction > loss_prediction_original):
            count += 1
            if count == 16:
                step_size = 0
            else:
                step_size = step_size / 2
            print('step_size=' + str(step_size))
            loss_prediction, x_connector, h_connector = \
                apply_and_check(isess, grads_and_vars,
                                step_size, data['u'][i], x_connector_current,
                                h_connector_current, data['y_true_float_corrected'][i])

        Wxx = isess.run(dr.Wxx)
        stable_flag = check_transition_matrix(Wxx)
        while not stable_flag:
            count += 1
            if count == 16:
                step_size = 0
            else:
                step_size = step_size / 2
            warnings.warn('not stable')
            print('step_size=' + str(step_size))
            Wxx = -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]
            stable_flag = check_transition_matrix(Wxx)
        isess.run([tf.assign(dr.Wxx, -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]),
                   tf.assign(dr.Wxxu[0],
                             -grads_and_vars[1][0] * step_size + grads_and_vars[1][1]),
                   tf.assign(dr.Wxu, -grads_and_vars[2][0] * step_size + grads_and_vars[2][1])])

        y_predicted_after_training = isess.run(dr.y_predicted,
                                               feed_dict={
                                                   dr.u_placeholder: data['u'][i],
                                                   dr.x_state_initial: x_connector_current,
                                                   dr.h_state_initial: h_connector_current,
                                                   dr.y_true: data['y_true_float_corrected'][i]
                                               })
        y_after_train.append(y_predicted_after_training)

        x_connector_current = x_connector
        h_connector_current = h_connector
        checking_loss.append(loss_prediction_original - loss_prediction)
        Wxx, Wxxu, Wxu = isess.run([dr.Wxx, dr.Wxxu[0], dr.Wxu])

        print(np.linalg.norm(data['y_true_float_corrected'][i].flatten()))
        print(checking_loss[-1])
        print(Wxx)
        print(Wxxu)
        # print(Wxx + Wxxu)
        print(Wxu)

print('optimization finished.')
# print(gradient_sum)
# print(grads_and_vars[0][1])
# print(grads_and_vars[1][1])
# print(grads_and_vars[2][1])
# print(loss_differences)

i = 0

y_predicted_before_training = isess.run(dr.y_predicted,
                                       feed_dict={
                                           dr.u_placeholder: data['u'][i],
                                           dr.x_state_initial: x_connectors[i],
                                           dr.h_state_initial: h_connectors[i],
                                           dr.y_true: data['y_true_float_corrected'][i],
                                           dr.Wxx: du.get('Wxx'),
                                           dr.Wxxu[0]: du.get('Wxxu')[0],
                                           dr.Wxu: du.get('Wxu')
                                       })

y_predicted_after_training = isess.run(dr.y_predicted,
                                       feed_dict={
                                           dr.u_placeholder: data['u'][i],
                                           dr.x_state_initial: x_connectors[i],
                                           dr.h_state_initial: h_connectors[i],
                                           dr.y_true: data['y_true_float_corrected'][i],
                                           dr.Wxx: Wxx,
                                           dr.Wxxu[0]: Wxxu,
                                           dr.Wxu: Wxu
                                       })

i = 0
plt.close()
plt.plot(y_before_train[i], '--')
plt.plot(y_after_train[i], '-.')
plt.plot(data['y_true_float_corrected'][i], alpha=0.5)
print(checking_loss[i])
i = i + 1

for i in range(len(y_before_train)):
    delta = np.array(y_before_train[i]) - np.array(y_after_train[i])
    norm = np.linalg.norm(delta.flatten())
    print(norm)
    print(checking_loss[i])
    print('\n')

'''
Wxx_hat = Wxx
Wxxu_hat = Wxxu
Wxu_hat = Wxu

du_hat = copy.deepcopy(du)
du_hat._secured_data['Wxx'] = Wxx_hat
du_hat._secured_data['Wxxu'] = [Wxxu_hat]
du_hat._secured_data['Wxu'] = Wxu_hat
parameter_package = du_hat.collect_parameter_for_x_scan()
du_hat._secured_data['x'] = du_hat.scan_x(parameter_package)
parameter_package = du_hat.collect_parameter_for_h_scan()
du_hat._secured_data['h'] = du_hat.scan_h(parameter_package)
parameter_package = du_hat.collect_parameter_for_y_scan()
du_hat._secured_data['y'] = du_hat.scan_y(parameter_package)

signal_length = tb.merge(data['u'], N_RECURRENT_STEP, DATA_SHIFT).shape[0]
index_range = list(range(signal_length))
plt.plot(du.get('y')[index_range])
plt.plot(du_hat.get('y')[index_range], '--')
# plt.plot(tb.merge(data['y_true_float_corrected'], N_RECURRENT_STEP, DATA_SHIFT)[index_range], '*', alpha=0.7)
print('mse x_hat vs x_true:' + str(tb.mse(du.get('y')[index_range], du_hat.get('y')[index_range])))

'''
