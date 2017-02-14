import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.ndimage

from DCM_RNN import toolboxes
importlib.reload(toolboxes)
from data import database_toolboxes as dbt
importlib.reload(dbt)
dbo = dbt.Operations()


os.chdir("/Users/yuanwang/Google_Drive/projects/Gits/DCM-RNN/data")
print('working directory is ' + os.getcwd())


# load data base
current_base_number = 0
current_base_name = 'DB' + str(current_base_number)
data_path = current_base_name + '.pkl'
data = dbo.load_database(data_path)

# check data
i = 0
du = toolboxes.DataUnit()
du.load_parameter_core(data[i])
du.recover_data_unit()
du.plot('y')


# for each DCM, recover fMRI signal with different t_delta [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
rMSEs = []
for i in range(1):
    print('current processing: ' + str(i))
    du = toolboxes.DataUnit()
    du.load_parameter_core(data[i])
    y_list = du.map('t_delta', [1. / 2 ** 8, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5], 'y')
    y_resampled = du.resample(y_list, y_list[-1].shape)
    rMSE = du.compare(y_resampled, y_resampled[0])
    rMSEs.append(rMSE)

for value in y_resampled:
    plt.plot(value[:, 2])


