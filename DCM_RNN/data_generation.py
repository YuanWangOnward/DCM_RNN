# Randomly generating seemingly realistic DCM model is time consuming.
# This file generates DCMs and store them for further analysis.
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from DCM_RNN import toolboxes


importlib.reload(toolboxes)

TOTAL_BASE_NUMBER = 1
SAMPLE_PER_BASE = 2

file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path + "/data")
print('working directory is ' + os.getcwd())


for current_base_number in range(0, TOTAL_BASE_NUMBER):
    current_base_name = 'DB' + str(current_base_number)
    current_data = []

    while len(current_data) < SAMPLE_PER_BASE:
        # new and setting
        du = toolboxes.DataUnit()
        du._secured_data['if_random_node_number'] = True
        du._secured_data['if_random_stimuli'] = True
        du._secured_data['if_random_x_state_initial'] = False
        du._secured_data['if_random_h_state_initial'] = False
        du._secured_data['t_delta'] = 0.25
        du._secured_data['t_scan'] = 5 * 60
        du._secured_data['learning_rate'] = 0.1
        du._secured_data['n_backpro'] = 12
        du.complete_data_unit(if_show_message=False)

        # add cores
        du_core = du.collect_parameter_core()
        current_data.append(du_core)
        print('Number of current cores base ' + str(len(current_data)))

    # sort cores
    current_data = sorted(current_data, key=lambda x: x.get('n_node'))

    # save cores
    print('Number of current cores base ' + str(len(current_data)))
    data_path = current_base_name + '.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(current_data, f)


'''
# Codes used to check generated data

# check cores base
current_base_number = 0
current_base_name = 'DB' + str(current_base_number)
data_path = current_base_name + '.pkl'
with open(data_path, 'rb') as f:
    para_core_loaded = pickle.load(f)


for i in range(0, 20):
    plt.figure()
    du = toolboxes.DataUnit()
    du.load_parameter_core(para_core_loaded[i])
    du.recover_data_unit()
    x_axis = np.arange(du._secured_data['n_time_point']) * du._secured_data['t_delta']
    n_node = du._secured_data['n_node']
    y = du._secured_data['y']
    plt.clf()
    for n in range(n_node):
        value = y[:, n].squeeze()
        plt.subplot(n_node, 1, n + 1)
        plt.plot(x_axis, value)
    #    time.sleep(0.5)


ii = 0
plt.clf()
plt.subplot(3,1,1)
plt.plot(x[:, ii])
plt.subplot(3,1,2)
x_demeaned = x - np.mean(x, axis=0)
plt.plot(x_demeaned[:, ii])
plt.subplot(3,1,3)
plt.plot(abs(fx[:, ii]))

x = du._secured_data['x']
x_demeaned = x - np.mean(x, axis=0)
fx = np.fft.fft(x_demeaned, axis=0) / np.sqrt(x.shape[0])
low_frequency_range = range(-int(x.shape[0]/200), int(x.shape[0]/200))
low_frequency_energy = np.linalg.norm(fx[low_frequency_range, :], axis = 0)
energy_persentage = low_frequency_energy / np.linalg.norm(fx, axis = 0)
print(energy_persentage)


# show results
        x_axis = np.arange(du._secured_data['n_time_point']) * du._secured_data['t_delta']
        n_node = du._secured_data['n_node']
        y = du._secured_data['y']
        plt.clf()
        for n in range(n_node):
            value = y[:, n].squeeze()
            plt.subplot(n_node, 1, n + 1)
            plt.plot(x_axis, value)


x = du._secured_data['x']
h = du._secured_data['h']
y = du._secured_data['y']

x_axis = np.arange(du._secured_data['n_time_point']) * du._secured_data['t_delta']
n_node = du._secured_data['n_node']

plt.clf()
for n in range(n_node):
    value = x[:, n]
    plt.subplot(n_node, 1, n + 1)
    plt.plot(x_axis, value)

plt.clf()
for n in range(n_node):
    value = h[:, n, :].squeeze()
    plt.subplot(n_node, 1, n + 1)
    plt.plot(x_axis, value)

plt.clf()
for n in range(n_node):
    value = y[:, n].squeeze()
    plt.subplot(n_node, 1, n + 1)
    plt.plot(x_axis, value)
'''