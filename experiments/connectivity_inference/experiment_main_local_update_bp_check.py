# add mask to gradient
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
import pandas as pd

MAX_EPOCHS = 16
CHECK_STEPS = 1
N_SEGMENTS = 128
N_RECURRENT_STEP = 128
# STEP_SIZE = 0.002 # for 32
# STEP_SIZE = 0.5
# STEP_SIZE = 0.001 # for 64
STEP_SIZE = 0.001 # 128
# STEP_SIZE = 0.0005 # for 256
DATA_SHIFT = int(N_RECURRENT_STEP / 4)
LEARNING_RATE = 0.01 / N_RECURRENT_STEP


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
dr.loss_weighting = {'prediction': 1., 'sparsity': 0.1, 'prior': 0.1, 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
dr.trainable_flags = {'Wxx': True,
                      'Wxxu': True,
                      'Wxu': True,
                      'alpha': True,
                      'E0': True,
                      'k': True,
                      'gamma': True,
                      'tao': True,
                      'epsilon': False,
                      'V0': False,
                      'TE': False,
                      'r0': False,
                      'theta0': False,
                      'x_h_coupling': False
                      }
dr.build_main_graph(neural_parameter_initial=neural_parameter_initial)

# process after building the main graph
mask = {dr.Wxx.name: np.ones((dr.n_region, dr.n_region)),
        dr.Wxxu[0].name: np.zeros((dr.n_region, dr.n_region)),
        dr.Wxu.name: np.zeros(dr.n_region).reshape(3, 1)}
mask[dr.Wxxu[0].name][2, 2] = 1
mask[dr.Wxu.name][0] = 1
dr.support_masks = dr.setup_support_mask(mask)
dr.x_parameter_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=dr.variable_scope_name_x_parameter)]
dr.trainable_variables_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]


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

N_TEST_SAMPLE = min(N_SEGMENTS, len(data['y_true_float_corrected']))

isess = tf.InteractiveSession()


def apply_and_check(isess, grads_and_vars, step_size, u, x_connector, h_connector, y_true):

    dr.update_variables_in_graph(isess, dr.trainable_variables_nodes,
                                 [-val[0] * step_size + val[1] for val in grads_and_vars])
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


'''
Wxx = np.array([[0.9576, 0.0069, -0.0097],
                [0.0496, 0.9375, 0.0252],
                [0.0182, 0.0415, 0.9458], ])
Wxxu = np.zeros((3, 3))
Wxxu[2, 2] = -0.0112
Wxu = np.array([0.0341, 0, 0]).reshape(3, 1)
'''
Wxx = np.identity(3) * (1 - 1.6 * du.get('t_delta'))
Wxxu = np.zeros((3, 3))
Wxu = np.array([0., 0., 0.]).reshape(3, 1)
dr.update_variables_in_graph(isess, dr.x_parameter_nodes, [Wxx, Wxxu, Wxu])


loss_differences = []
loss_totals = []
y_before_train = []
y_after_train = []
x_connectors = []
h_connectors = []
gradients = []
step_sizes = []
for epoch in range(MAX_EPOCHS):
    x_connector_current = dr.set_initial_neural_state_as_zeros(dr.n_region)
    h_connector_current = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region)
    for i in range(len(data['y_true_float_corrected'])):
        print('current processing ' + str(i))

        grads_and_vars, x_connector, h_connector, loss_prediction, loss_total, y_predicted_before_training = \
            isess.run([dr.grads_and_vars, dr.x_connector, dr.h_connector,
                       dr.loss_prediction, dr.loss_total, dr.y_predicted],
                      feed_dict={
                          dr.u_placeholder: data['u'][i],
                          dr.x_state_initial: x_connector_current,
                          dr.h_state_initial: h_connector_current,
                          dr.y_true: data['y_true_float_corrected'][i]
                      })

        # apply mask to gradients
        variable_names = [v.name for v in tf.trainable_variables()]
        for idx in range(len(grads_and_vars)):
            variable_name = variable_names[idx]
            grads_and_vars[idx] = (grads_and_vars[idx][0] * dr.support_masks[variable_name], grads_and_vars[idx][1])

        # logs
        loss_prediction_original = loss_prediction
        gradients.append(grads_and_vars)
        y_before_train.append(y_predicted_before_training)
        x_connectors.append(x_connector_current)
        h_connectors.append(h_connector_current)
        loss_totals.append(loss_total)

        # updating with back-tracking
        step_size = STEP_SIZE
        loss_prediction, x_connector, h_connector = \
            apply_and_check(isess, grads_and_vars, step_size, data['u'][i],
                            x_connector_current,
                            h_connector_current,
                            data['y_true_float_corrected'][i])

        count = 0
        Wxx = isess.run(dr.Wxx)
        stable_flag = check_transition_matrix(Wxx)
        while not stable_flag:
            count += 1
            if count == 20:
                step_size = 0
            else:
                step_size = step_size / 2
            warnings.warn('not stable')
            print('step_size=' + str(step_size))
            Wxx = -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]
            stable_flag = check_transition_matrix(Wxx)

        while (loss_prediction > loss_prediction_original or np.isnan(loss_prediction)):
            count += 1
            if count == 20:
                step_size = 0.
            else:
                step_size = step_size / 2
            print('step_size=' + str(step_size))
            loss_prediction, x_connector, h_connector = \
                apply_and_check(isess, grads_and_vars,
                                step_size, data['u'][i], x_connector_current,
                                h_connector_current, data['y_true_float_corrected'][i])
            if step_size == 0.0:
                break

        dr.update_variables_in_graph(isess, dr.trainable_variables_nodes,
                                     [-val[0] * step_size + val[1] for val in grads_and_vars])

        y_predicted_after_training = isess.run(dr.y_predicted,
                                               feed_dict={
                                                   dr.u_placeholder: data['u'][i],
                                                   dr.x_state_initial: x_connector_current,
                                                   dr.h_state_initial: h_connector_current,
                                                   dr.y_true: data['y_true_float_corrected'][i]
                                               })
        y_after_train.append(y_predicted_after_training)
        step_sizes.append(step_size)
        x_connector_current = x_connector
        h_connector_current = h_connector
        loss_differences.append(loss_prediction_original - loss_prediction)
        Wxx, Wxxu, Wxu = isess.run([dr.Wxx, dr.Wxxu[0], dr.Wxu])

        print(loss_totals[-1])
        print(loss_differences[-1])
        print(Wxx)
        print(Wxxu)
        # print(Wxx + Wxxu)
        print(Wxu)

print('optimization finished.')



# check gradients
grad_sum_wxx = sum([val[0][0] for val in gradients])
grad_sum_wxxu = sum([val[1][0] for val in gradients])
grad_sum_wxu = sum([val[2][0] for val in gradients])
print('Wxx')
print('Wxxu')
print('Wxu')


# visually check y in each segmentation, single node
i = 0
plt.close()
plt.plot([val[0] for val in y_before_train[i]], '--', label='y_hat_before_training')
plt.plot([val[0] for val in y_after_train[i]], '-.', label='y_hat_after_training')
plt.plot([val[0] for val in data['y_true_float_corrected'][i]], alpha=0.5, label='y_true')
plt.legend(loc=0)
plt.xlabel('time index')
plt.ylabel('value')
print(loss_differences[i])
i = i + 1


# visually check y in each segmentation, three nodes
i = 0
plt.close()
plt.plot(y_before_train[i], '--', label='y_hat_before_training')
plt.plot(y_after_train[i], '-.', label='y_hat_after_training')
plt.plot(data['y_true_float_corrected'][i], alpha=0.5, label='y_true')
plt.legend()
print(loss_differences[i])
i = i + 1


# check loss
df = pd.DataFrame()
df['loss_total'] = loss_totals
df['loss_delta'] = loss_differences
df['improvement_persentage'] = df['loss_delta'] / df['loss_total']
x_axis = range(len(df))
plt.bar(x_axis, df['loss_total'], label='loss_total')
plt.bar(x_axis, df['loss_delta'], label='loss_detal', color='green')
plt.legend()
# filename = '/Users/yuanwang/Desktop/DCM_RNN_progress/BP.csv'
# df.to_csv(filename)


print(du.get('Wxx'))
print(du.get('Wxxu'))
print(du.get('Wxu'))

plt.hist(step_sizes, bins=128)
