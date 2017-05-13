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
dr.learning_rate = tm.LEARNING_RATE
dr.shift_data = tm.DATA_SHIFT
dr.n_recurrent_step = tm.N_RECURRENT_STEP
dr.loss_weighting['smooth'] = tm.SMOOTH_WEIGHT
if tm.IF_NODE_MODE:
    dr.n_region = 1
for key in dr.trainable_flags.keys():
    # in the initialization graph, the hemodynamic parameters are not trainable
    dr.trainable_flags[key] = False
dr.build_an_initializer_graph()
print('Building tf model done.')

# prepare data
data = tm.prepare_data(du, dr)
print('Preparing data done.')
