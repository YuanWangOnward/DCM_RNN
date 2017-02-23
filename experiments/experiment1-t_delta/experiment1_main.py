import importlib
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from DCM_RNN import toolboxes

importlib.reload(toolboxes)
from DCM_RNN import database_toolboxes as dbt
importlib.reload(dbt)
dbo = dbt.Operations()


os.chdir("/Users/yuanwang/Google_Drive/projects/Gits/DCM-RNN/data")
print('working directory is ' + os.getcwd())

# load cores base
current_base_number = 2
current_base_name = 'DB' + str(current_base_number)
data_path = current_base_name + '.pkl'
cores = dbo.load_database(data_path)

# check cores 13, 27, 68
i = 13
du = toolboxes.DataUnit()
du.load_parameter_core(cores[i])
print(du._secured_data['n_node'])
du.recover_data_unit()
du.plot('y')


# for each DCM, recover fMRI signal with different t_delta
t_delta_list = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]
rMSEs = []
for i in range(len(cores)):
    print('current processing: ' + str(i))
    du = toolboxes.DataUnit()
    du.load_parameter_core(cores[i])
    y_list = du.map('t_delta', t_delta_list, 'y')
    y_resampled = du.resample(y_list, y_list[-1].shape)
    rMSE = du.compare(y_resampled, y_resampled[0])
    rMSEs.append(rMSE)
    if np.mod(i, 20) == 19:
        data_path = os.getcwd() + '/../experiments/experiment1-t_delta/results' + str(i) + '.pkl'
        with open(data_path, 'wb') as f:
            pickle.dump(rMSEs, f)

for value in y_resampled:
    plt.plot(value[:, 0])

# load result
'''
data_path = os.getcwd() + '/../experiments/experiment1-t_delta/results' + str(i) + '.pkl'
with open(data_path, 'rb') as f:
    rMSEs = pickle.load(f)
'''

# plot histogram of rMSE

rMSEs = np.array(rMSEs)
histogram = plt.figure()
bins = np.linspace(0, 1, 100)
for n in range(rMSEs.shape[1]):
    temp = rMSEs[:, n]
    temp = temp[~np.isnan(temp)]
    plt.hist(temp, bins, alpha=0.5)
plt.show()
