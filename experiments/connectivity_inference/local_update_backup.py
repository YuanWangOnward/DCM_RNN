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

MAX_EPOCHS = 16
CHECK_STEPS = 1
N_RECURRENT_STEP = 128
DATA_SHIFT = int(N_RECURRENT_STEP / 4)
MAX_BACK_TRACK = 8
MAX_CHANGE = 0.001

print(os.getcwd())
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)
MAX_SEGMENTS = mth.ceil(len(du.get('y')) - N_RECURRENT_STEP / DATA_SHIFT)

# specify initialization values, loss weighting factors, and mask (support of effective connectivity)
x_parameter_initial = {}
x_parameter_initial['A'] = - np.eye(du.get('n_node'))
x_parameter_initial['B'] = [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))]
x_parameter_initial['C'] = np.zeros((du.get('n_node'), du.get('n_stimuli')))
x_parameter_initial['C'][0, 0] = 1

h_parameter_inital = du.get_standard_hemodynamic_parameters(du.get('n_node'))
h_parameter_inital['x_h_coupling'] = 1.

loss_weighting = {'prediction': 1., 'sparsity': 0.1, 'prior': 0.1, 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}

mask = {
    'Wxx': np.ones((du.get('n_node'), du.get('n_node'))),
    'Wxxu': [np.zeros((du.get('n_node'), du.get('n_node'))) for _ in range(du.get('n_stimuli'))],
    'Wxu': np.zeros((du.get('n_node'), du.get('n_stimuli')))
}
mask['Wxxu'][0][2, 2] = 1
mask['Wxu'][0, 0] = 1

du_hat = du.initialize_a_training_unit(x_parameter_initial['A'],
                                       x_parameter_initial['B'],
                                       x_parameter_initial['C'],
                                       np.array(h_parameter_inital))



# build tensorflow model
print('building model')
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
dr.max_back_track_steps = MAX_BACK_TRACK
dr.max_parameter_change_per_iteration = MAX_CHANGE
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
dr.loss_weighting = loss_weighting
dr.build_main_graph(neural_parameter_initial=x_parameter_initial)

# process after building the main graph
dr.support_masks = dr.setup_support_mask(mask)
dr.x_parameter_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=dr.variable_scope_name_x_parameter)]
dr.trainable_variables_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
dr.trainable_variables_names = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
du_hat.variable_names_in_graph = du_hat.parse_variable_names(dr.trainable_variables_names)


# prepare data for training
data = {
    'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
    'x_initial': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
    'h_initial': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
    'x_whole': tb.split(du.get('x'), n_segment=dr.n_recurrent_step + dr.shift_u_x, n_step=dr.shift_data, shift=0),
    'h_whole': tb.split(du.get('h'), n_segment=dr.n_recurrent_step + dr.shift_x_y, n_step=dr.shift_data, shift=1),
    'h_predicted': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
    'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}
for k in data.keys():
    data[k] = data[k][: MAX_SEGMENTS]

# start session
isess = tf.InteractiveSession()
isess.run(tf.global_variables_initializer())
dr.update_variables_in_graph(isess, dr.x_parameter_nodes,
                             [x_parameter_initial['A']] + x_parameter_initial['B'] + [x_parameter_initial['C']])

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



print('start inference')
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
    for i in range(len(data['y'])):
        print('current processing ' + str(i))

        grads_and_vars, x_connector, h_connector, loss_prediction, loss_total, y_predicted_before_training = \
            isess.run([dr.grads_and_vars, dr.x_connector, dr.h_connector,
                       dr.loss_prediction, dr.loss_total, dr.y_predicted],
                      feed_dict={
                          dr.u_placeholder: data['u'][i],
                          dr.x_state_initial: x_connector_current,
                          dr.h_state_initial: h_connector_current,
                          dr.y_true: data['y'][i]
                      })

        # apply mask to gradients
        variable_names = [v.name for v in tf.trainable_variables()]
        for idx in range(len(grads_and_vars)):
            variable_name = variable_names[idx]
            grads_and_vars[idx] = (grads_and_vars[idx][0] * dr.support_masks[variable_name], grads_and_vars[idx][1])

        # logs
        gradients.append(grads_and_vars)
        y_before_train.append(y_predicted_before_training)
        x_connectors.append(x_connector_current)
        h_connectors.append(h_connector_current)
        loss_totals.append(loss_total)

        # record the loss before applying gradient
        du_hat.update_trainable_variables(grads_and_vars, step_size=0)
        du_hat.regenerate_data(data['u'][i])
        y_hat_original = du_hat.get('y')
        loss_prediction_original = tb.mse(y_hat_original, du.get('y'))

        # adaptive step size, making max value change dr.max_parameter_change_per_iteration
        max_gradient = max([np.max(np.abs(np.array(g[0])).flatten()) for g in grads_and_vars])
        STEP_SIZE = dr.max_parameter_change_per_iteration / max_gradient
        print('max gradient:    ' + str(max_gradient))
        print('STEP_SIZE:    ' + str(STEP_SIZE))

        # try to find proper step size
        step_size = STEP_SIZE
        count = 0
        du_hat.update_trainable_variables(grads_and_vars, step_size)
        
        step_size = STEP_SIZE
        loss_prediction, x_connector, h_connector = \
            apply_and_check(isess, grads_and_vars, step_size, data['u'][i],
                            x_connector_current,
                            h_connector_current,
                            data['y'][i])

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
                                h_connector_current, data['y'][i])
            if step_size == 0.0:
                break

        dr.update_variables_in_graph(isess, dr.trainable_variables_nodes,
                                     [-val[0] * step_size + val[1] for val in grads_and_vars])

        y_predicted_after_training = isess.run(dr.y_predicted,
                                               feed_dict={
                                                   dr.u_placeholder: data['u'][i],
                                                   dr.x_state_initial: x_connector_current,
                                                   dr.h_state_initial: h_connector_current,
                                                   dr.y_true: data['y'][i]
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
plt.plot([val[0] for val in data['y'][i]], alpha=0.5, label='y_true')
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
plt.plot(data['y'][i], alpha=0.5, label='y_true')
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
