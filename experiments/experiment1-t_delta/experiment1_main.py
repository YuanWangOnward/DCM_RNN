import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle

from DCM_RNN import toolboxes
importlib.reload(toolboxes)
from data import database_toolboxes as dbt
importlib.reload(dbt)
dbo = dbt.Operations()


# load data base
current_base_number = 0
current_base_name = 'data/DB' + str(current_base_number)
data_path = current_base_name + '.pkl'
data = dbo.load_database(data_path)

# check data
i = 0
du = toolboxes.DataUnit()
du.load_parameter_core(data[i])
du.recover_data_unit()
du.plot('y')

# for each DCM, recover fMRI signal with different t_delta [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
i = 0



# compare and record MSE
# plot the result




sc = toolboxes.Scanner()
du = toolboxes.DataUnit()

du._secured_data['if_random_node_number'] = False
# du._secured_data['if_random_delta_t'] = True
# du._secured_data['if_random_scan_time'] = True
du._secured_data['if_random_stimuli'] = True
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False

du._secured_data['t_delta'] = 0.25
du._secured_data['t_scan'] = 5 * 60
du._secured_data['n_node'] = 3
du._secured_data['learning_rate'] = 0.1
du._secured_data['n_backpro'] = 12
du.complete_data_unit(if_show_message=False)


x = du._secured_data['x']
h = du._secured_data['h']
y = du._secured_data['y']


import matplotlib
print(matplotlib.rcParams['backend'])


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

