# This code is to generate resource data.

import os
import pickle
import numpy as np
from dcm_rnn import toolboxes
import matplotlib.pyplot as plt

IF_CHECK_RESULT = True
IF_LOADING_TEST = True

file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path + "/../")
print('working directory is ' + os.getcwd())

# template 0, used in ISMRM2017 abstract
# new and setting
du = toolboxes.DataUnit()
du._secured_data['if_random_neural_parameter'] = False
du._secured_data['if_random_hemodynamic_parameter'] = False
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False
du._secured_data['if_random_stimuli'] = True
du._secured_data['if_random_node_number'] = False
du._secured_data['if_random_stimuli_number'] = False
du._secured_data['if_random_delta_t'] = False
du._secured_data['if_random_scan_time'] = False
du._secured_data['t_delta'] = 1. / 16.
du._secured_data['t_scan'] = 3 * 60
du._secured_data['n_node'] = 3
du._secured_data['n_stimuli'] = 1
du._secured_data['A'] = np.array([[-1, 0, 0],
                                  [0.8, -1, 0.4],
                                  [0.4, 0.8, -1]])
du._secured_data['B'] = [np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, -0.4]])]
du._secured_data['C'] = np.array([0.4, 0, 0]).reshape(3, 1)
du._secured_data['learning_rate'] = 0.1
du._secured_data['n_backpro'] = 12
du.complete_data_unit(if_show_message=False, if_check_property=False)

if IF_CHECK_RESULT:
    x = du._secured_data['x']
    h = du._secured_data['h']
    y = du._secured_data['y']

    x_axis = np.arange(du._secured_data['n_time_point']) * du._secured_data['t_delta']
    n_node = du._secured_data['n_node']

    plt.figure()
    for n in range(n_node):
        value = x[:, n]
        plt.subplot(n_node, 1, n + 1)
        plt.plot(x_axis, value)

    plt.figure()
    for n in range(n_node):
        value = h[:, n, :].squeeze()
        plt.subplot(n_node, 1, n + 1)
        plt.plot(x_axis, value)

    plt.figure()
    for n in range(n_node):
        value = y[:, n].squeeze()
        plt.subplot(n_node, 1, n + 1)
        plt.plot(x_axis, value)

# save
data_path = 'dcm_rnn/resources/template0' + '.pkl'
with open(data_path, 'wb') as f:
    pickle.dump(du, f)

if IF_LOADING_TEST:
    with open(data_path, 'rb') as f:
        du_load = pickle.load(f)
        np.testing.assert_array_equal(du._secured_data['y'], du_load._secured_data['y'])

# template 1, randomly generated
du = toolboxes.DataUnit()
du._secured_data['if_random_node_number'] = False
du._secured_data['if_random_stimuli'] = True
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False
du._secured_data['t_delta'] = 1. /16.
du._secured_data['t_scan'] = 3 * 60
du._secured_data['learning_rate'] = 0.1
du._secured_data['n_backpro'] = 12
du._secured_data['n_node'] = 3
du._secured_data['n_stimuli'] = 1
du.complete_data_unit(if_show_message=False, if_check_property=True)

if IF_CHECK_RESULT:
    x = du._secured_data['x']
    h = du._secured_data['h']
    y = du._secured_data['y']

    x_axis = np.arange(du._secured_data['n_time_point']) * du._secured_data['t_delta']
    n_node = du._secured_data['n_node']

    plt.figure()
    for n in range(n_node):
        value = x[:, n]
        plt.subplot(n_node, 1, n + 1)
        plt.plot(x_axis, value)

    plt.figure()
    for n in range(n_node):
        value = h[:, n, :].squeeze()
        plt.subplot(n_node, 1, n + 1)
        plt.plot(x_axis, value)

    plt.figure()
    for n in range(n_node):
        value = y[:, n].squeeze()
        plt.subplot(n_node, 1, n + 1)
        plt.plot(x_axis, value)

# save
data_path = 'dcm_rnn/resources/template1' + '.pkl'
with open(data_path, 'wb') as f:
    pickle.dump(du, f)

if IF_LOADING_TEST:
    with open(data_path, 'rb') as f:
        du_load = pickle.load(f)
        np.testing.assert_array_equal(du._secured_data['y'], du_load._secured_data['y'])



