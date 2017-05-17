import sys
sys.path.append('dcm_rnn')
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_model as tfm
import toolboxes as tb
import numpy as np
import os
import pickle
import datetime
import warnings

import random
import training_manager
import multiprocessing
from multiprocessing.pool import Pool
import itertools


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


# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
elif '/share/apps/python3/' in sys.executable:
    PROJECT_DIR = '/home/yw1225/projects/DCM_RNN'
    print("It seems a remote run on NYU HPC")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
else:
    PROJECT_DIR = '.'
    print("Not sure executing machine. Make sure to set PROJECT_DIR properly.")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)

LOCAL_DEBUGGING = True

tm = training_manager.TrainingManager()
if LOCAL_DEBUGGING is True:
    tm.N_RECURRENT_STEP = 4
    tm.N_SEGMENTS = 8
    tm.MAX_EPOCHS = 1
    tm.MAX_EPOCHS_INNER = 2
    tm.CHECK_STEPS = 4
    tm.IF_NODE_MODE = False
    tm.N_PACKAGES = 2
else:
    tm.IF_NODE_MODE = False
    tm.N_PACKAGES = 1

    tm.IF_RANDOM_H_PARA = False
    tm.IF_RANDOM_H_STATE_INIT = False
    tm.IF_NOISED_Y = False


# load in data
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)
print('Loading data done.')

# create and configure DcmRnn instance
dr = tfm.DcmRnn()
dr.collect_parameters(du)
tm.prepare_dcm_rnn(dr, tag='initializer')
# dr.build_an_initializer_graph()
print('Creating and configuring tf model done.')

# get distributed data data_package
configure_package = tm.prepare_distributed_configure_package()
print('Preparing distributed data data_package done.')

# modify each data_package according to each particular experimental case, store in a list
package_list = tm.modify_configure_packages(configure_package, 'SNR', range(2, 2 + tm.N_PACKAGES))
# package_list = configure_package

# start parallel processing
cpu_count = multiprocessing.cpu_count()
tm.N_CORES = min(cpu_count, len(package_list))
print('There are ' + str(cpu_count) + ' cores available. ' + str(tm.N_CORES) + ' fo them are used.')
iterator = itertools.product(*[[du], [dr], package_list])
with Pool(tm.N_CORES) as p:
    # prepare data
    package_list = p.starmap(tm.prepare_data, iterator)

    # modify data if necessary
    # iterator = itertools.product(*[package_list, ['H_PARA_INITIAL'], [values]])
    # package_list = p.starmap(tm.modify_signel_data_package, iterator)

    # build graph and training must be done in one function
    iterator = itertools.product(*[[dr], package_list])
    package_list = p.starmap(tm.build_initializer_graph_and_train, iterator)

data = package_list[0].data
assert 'loss_smooth_normalizer' in data.keys()

signal_length = data['x_true_merged'].data.shape[0]
check_length = tm.N_SEGMENTS * tm.DATA_SHIFT


# test split and merge
recovered = tb.merge(tb.split(du.get('x'), 64, 4), 64, 4)
original = du.get('x')[:len(recovered)]
np.testing.assert_array_almost_equal(recovered, original)

# check recorded u and processed u
original = du.get('u')[:check_length]
recovered = data['u_merged'][:check_length]
np.testing.assert_array_almost_equal(recovered, original)
plt.plot(original, '-', label="original")
plt.plot(recovered, '*', label="processed")
plt.show()

# check recorded x_true and the original x_true
original = du.get('u')[:check_length]
recovered = data['u_merged'][:check_length]
np.testing.assert_array_almost_equal(recovered, original)
plt.plot(du.get('x')[:check_length], label="original")
plt.plot(data['x_true_merged'][:check_length], label="processed")
plt.show()



# check recorded x_true and x_hat
plt.plot(data['x_true_merged'][:128], label="original")
plt.plot(data['x_hat_merged'].data[:128], label="original")
plt.show()

# check recorded x_true and the original u
plt.plot(data['x_true_merged'][:128])
plt.plot(du.get('u')[:128] / 4)
plt.show()

# check recorded x_true and the original u
plt.plot(data['x_true_merged'].data[:128])
plt.plot(du.get('u')[:128] / 4)
plt.show()
