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



LOCAL_DEBUGGING = False

tm = training_manager.TrainingManager()
if LOCAL_DEBUGGING is True:
    tm.N_RECURRENT_STEP = 4
    tm.N_SEGMENTS = 64
    tm.MAX_EPOCHS = 1
    tm.MAX_EPOCHS_INNER = 4
    tm.CHECK_STEPS = 4
    tm.IF_NODE_MODE = True
    tm.N_PACKAGES = 2
else:
    tm.IF_NODE_MODE = False
    tm.N_PACKAGES = 1
    tm.N_SEGMENTS = 256

    tm.IF_RANDOM_H_PARA = False
    tm.IF_RANDOM_H_STATE_INIT = False
    tm.IF_NOISED_Y = False

# load in spm_data
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)
print('Loading spm_data done.')

# create and configure DcmRnn instance
dr = tfm.DcmRnn()
dr.collect_parameters(du)
tm.prepare_dcm_rnn(dr, tag='initializer')
# dr.build_an_initializer_graph()
print('Creating and configuring tf model done.')

# get distributed spm_data data_package
configure_package = tm.prepare_distributed_configure_package()
print('Preparing distributed spm_data data_package done.')

# modify each data_package according to each particular experimental case, store in a list
if tm.IF_NOISED_Y:
    package_list = tm.modify_configure_packages(configure_package, 'SNR', range(2, 2 + tm.N_PACKAGES))
else:
    package_list = configure_package

# package_list = configure_package

# start parallel processing
cpu_count = multiprocessing.cpu_count()
tm.N_CORES = min(cpu_count, len(package_list))
print('There are ' + str(cpu_count) + ' cores available. ' + str(tm.N_CORES) + ' fo them are used.')
iterator = itertools.product(*[[du], [dr], package_list])
with Pool(tm.N_CORES) as p:
    # prepare spm_data
    package_list = p.starmap(tm.prepare_data, iterator)

    # modify spm_data if necessary
    iterator = itertools.product(*[package_list, ['du'], [du]])
    package_list = p.starmap(tm.modify_signel_data_package, iterator)

    # build graph and training must be done in one function
    iterator = itertools.product(*[[dr], package_list])
    package_list = p.starmap(tm.build_initializer_graph_and_train, iterator)

# collect results
