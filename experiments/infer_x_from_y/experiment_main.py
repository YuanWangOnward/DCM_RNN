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
import sys
import random
import training_manager
import multiprocessing
from multiprocessing.pool import Pool
import itertools
from itertools import product


# global setting, you need to modify it accordingly
# project directory
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


tm = training_manager.TrainingManager()
tm.N_RECURRENT_STEP = 8
tm.IF_NODE_MODE = True


# load in data
data_path = PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)
print('Loading data done.')

# build model
dr = tfm.DcmRnn()
dr.collect_parameters(du)
tm.prepare_dcm_rnn(dr, tag='initializer')
# dr.build_an_initializer_graph()
print('Building tf model done.')

# get distributed data package
package = tm.prepare_distributed_data_package()
print('Preparing distributed data package done.')

# modify each package according to each particular experimental case, store in a list
package_list = [package, package, package, package]

# prepare data in parallel
cpu_count = multiprocessing.cpu_count()
print('There are ' + str(cpu_count) + ' cores available.')
iterator = list(itertools.product(*[[du], [dr], package_list]))
with Pool(cpu_count) as p:
    p.starmap(tm.prepare_data, iterator)
# print('Preparing data done for process ' + str(multiprocessing.current_process()))


# training
# tm.train(dr, dr, dr)
# print('Training done.')