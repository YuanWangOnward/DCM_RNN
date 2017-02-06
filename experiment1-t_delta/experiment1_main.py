import importlib
import numpy as np
import matplotlib.pyplot as plt

from DCM_RNN import toolboxes
importlib.reload(toolboxes)

sc = toolboxes.Scanner()
du = toolboxes.DataUnit()

# du._secured_data['if_random_node_number'] = True
# du._secured_data['if_random_delta_t'] = True
# du._secured_data['if_random_scan_time'] = True
du._secured_data['if_random_stimuli'] = True

du._secured_data['n_node'] = 3
du._secured_data['t_delta'] = 0.25
du._secured_data['t_scan'] = 5 * 60

du._secured_data['learning_rate'] = 0.1
du._secured_data['n_backpro'] = 12
du.complete_data_unit(if_show_message=True)

# print(du._secured_data.keys())
x_connection_matrices = du.get_dcm_rnn_x_matrices()
x_state_initial = du._secured_data['initial_x_state']
u = du._secured_data['u']
#x = sc.scan_x(x_connection_matrices, x_state_initial, u)
x = du._secured_data['x']
print(u.shape)
print(x.shape)

import matplotlib
print(matplotlib.rcParams['backend'])


x_axis = np.arange(du._secured_data['n_time_point']) * du._secured_data['t_delta']
n_node = du._secured_data['n_node']
# plt.ion()
# plt.interactive(True)

plt.clf()
for n in range(n_node):
    y = x[:, n]
    plt.subplot(n_node, 1, n + 1)
    plt.plot(x_axis, y)
A = x_connection_matrices[0]
B = x_connection_matrices[1]
C = x_connection_matrices[2]
w, v = np.linalg.eig(A)
print(str(max(w.real)))
print(np.var(x,0))
print(sc.if_proper_x(x))
