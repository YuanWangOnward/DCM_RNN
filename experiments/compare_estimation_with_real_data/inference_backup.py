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
from scipy.interpolate import interp1d
import importlib
from scipy.fftpack import idct, dct

DATA_DIR = os.path.join(PROJECT_DIR, 'dcm_rnn', 'resources', 'SPM_data_attention.pkl')
MAX_EPOCHS = 32 * 3
CHECK_STEPS = 1
N_RECURRENT_STEP = 192
DATA_SHIFT = 2
MAX_BACK_TRACK = 16
MAX_CHANGE = 0.001
BATCH_RANDOM_DROP_RATE = 1.
TARGET_T_DELTA = 1. / 16
N_CONFOUNDS = 19

du = tb.DataUnit()
# load SPM attention spm_data from resources
spm_data = pickle.load(open(DATA_DIR, 'rb'))

# process spm_data,
# up sample u and y to 16 frame/second
shape = list(spm_data['u'].shape)
shape[0] = int(spm_data['TR'] / TARGET_T_DELTA * shape[0])
spm_data['u_upsampled'] = du.resample(spm_data['u'], shape, order=3)
spm_data['y_upsampled'] = du.resample(spm_data['y'], shape, order=3)

# assume the observation model is
# y = DCM_RNN(u) + Confounds * weights + noise
# Confounds are the first cosine transfer basis (19)
n_time_point = spm_data['u_upsampled'].shape[0]
confounds = idct(np.eye(n_time_point)[:, :N_CONFOUNDS], axis=0, norm='ortho')
# plt.plot(confounds)


# shift y's so that they are around 0 when there is no input
t_total = spm_data['TR'] * spm_data['y'].shape[0]
t_axis_original = np.linspace(0, t_total, t_total / spm_data['TR'], endpoint=True)
t_axis_new = np.linspace(0, t_total, t_total / TARGET_T_DELTA, endpoint=True)
spm_data['y_upsampled'] = np.zeros((len(t_axis_new), spm_data['y'].shape[1]))
for i in range(spm_data['y'].shape[1]):
    interpolate_func = interp1d(t_axis_original, spm_data['y'][:, i], kind='cubic')
    spm_data['y_upsampled'][:, i] = interpolate_func(t_axis_new)
spm_data['u_upsampled'] = np.zeros((len(t_axis_new), spm_data['u'].shape[1]))
for i in range(spm_data['u'].shape[1]):
    interpolate_func = interp1d(t_axis_original, spm_data['u'][:, i])
    spm_data['u_upsampled'][:, i] = interpolate_func(t_axis_new)
temporal_mask = spm_data['u_upsampled'][:, 0] < 0.5
temp = copy.deepcopy(spm_data['y_upsampled'])
mean_values = np.mean(temp[temporal_mask, :], axis=0)
mean_values = np.repeat(mean_values.reshape([1, 3]), spm_data['y_upsampled'].shape[0], axis=0)
spm_data['y_upsampled'] = spm_data['y_upsampled'] - mean_values


# build dcm_rnn model
importlib.reload(tfm)
tf.reset_default_graph()
dr = tfm.DcmRnn()
dr.n_region = 3
dr.t_delta = TARGET_T_DELTA
dr.n_stimuli = 3
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
dr.loss_weighting = {'prediction': 1., 'sparsity': 0.1, 'prior': 0.1, 'Wxx': 1., 'Wxxu': 0., 'Wxu': 0.}
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
A = np.eye(dr.n_region) * (-0.01)
B = [np.zeros((dr.n_region, dr.n_region))] * dr.n_region
C = np.zeros((dr.n_region, dr.n_stimuli))
C[0, 0] = 0
neural_parameter_initial = {'A': A, 'B': B, 'C': C}
hemodynamic_parameter_initial = dr.get_standard_hemodynamic_parameters(dr.n_region).astype(np.float32)
hemodynamic_parameter_initial['x_h_coupling'] = 1.
dr.build_main_graph(neural_parameter_initial=neural_parameter_initial,
                    hemodynamic_parameter_initial=hemodynamic_parameter_initial)

# process after building the main graph
mask = {dr.Wxx.name: np.ones((dr.n_region, dr.n_region)),
        dr.Wxxu[0].name: np.zeros((dr.n_region, dr.n_region)),
        dr.Wxxu[1].name: np.zeros((dr.n_region, dr.n_region)),
        dr.Wxxu[2].name: np.zeros((dr.n_region, dr.n_region)),
        dr.Wxu.name: np.zeros((dr.n_region, dr.n_stimuli)).reshape(dr.n_region, dr.n_stimuli)}
mask[dr.Wxx.name][0, 2] = 0
mask[dr.Wxx.name][2, 0] = 0
mask[dr.Wxxu[1].name][1, 0] = 1
mask[dr.Wxxu[2].name][0, 1] = 1
mask[dr.Wxu.name][0, 0] = 1
dr.support_masks = dr.setup_support_mask(mask)
dr.x_parameter_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=dr.variable_scope_name_x_parameter)]
dr.trainable_variables_nodes = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

# prepare spm_data for training
data = {
    'u': tb.split(spm_data['u_upsampled'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
    'y': tb.split(spm_data['y_upsampled'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
}
for k in data.keys():
    data[k] = data[k][: N_SEGMENTS]
N_TEST_SAMPLE = min(N_SEGMENTS, len(data['y']))


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

isess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=4))
isess.run(tf.global_variables_initializer())

# isess.run(tf.assign(dr.Wxx, np.zeros((dr.n_region, dr.n_region))))

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
        Wxx, Wxxu, Wxu = isess.run([dr.Wxx, dr.Wxxu, dr.Wxu])

        print(loss_totals[-1])
        print(loss_differences[-1])
        print(Wxx)
        print(Wxxu)
        # print(Wxx + Wxxu)
        print(Wxu)

print('optimization finished.')


du_hat = tb.DataUnit()
du_hat._secured_data['n_node'] = 3
du_hat._secured_data['n_stimuli'] = 3
Wxx = np.zeros((dr.n_region, dr.n_region))
Wxx = np.eye(dr.n_region) * -0.02
du_hat._secured_data['A'] = (Wxx - np.eye(du_hat.get('n_node'))) / dr.t_delta
du_hat._secured_data['B'] = [m / dr.t_delta for m in Wxxu]
Wxu[0, 0] = 0.005
du_hat._secured_data['C'] = Wxu / dr.t_delta
du_hat._secured_data['t_delta'] = dr.t_delta
du_hat._secured_data['u'] = spm_data['u_upsampled']
du_hat._secured_data['t_scan'] = len(spm_data['u_upsampled']) * dr.t_delta
hemodynamic_parameter = du_hat.get_standard_hemodynamic_parameters(dr.n_region)
temp = isess.run(dr.h_parameters)
for c in range(temp.shape[1]):
    hemodynamic_parameter[hemodynamic_parameter.columns[c]] = temp[:, c]
du_hat._secured_data['hemodynamic_parameter'] = hemodynamic_parameter
du_hat._secured_data['initial_x_state'] = du_hat.set_initial_neural_state_as_zeros(dr.n_region)
du_hat._secured_data['initial_h_state'] = du_hat.set_initial_hemodynamic_state_as_inactivated(dr.n_region)
du_hat.complete_data_unit(start_categorty=2, if_check_property=False, if_show_message=True)

i = 0
plt.plot(spm_data['y_upsampled'][:, i]);
plt.plot(spm_data['u_upsampled'][:, i], '--');
plt.plot(du_hat.get('y')[:, i]);
plt.plot(du_hat.get('x')[:, i], '.-');

plt.plot(du_hat.get('x')[500:1000, 0]);
plt.plot(du_hat.get('h')[800:820, 0, :]);




