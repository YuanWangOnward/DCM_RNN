import tensorflow as tf
import dcm_rnn.tf_model as tfm
import dcm_rnn.toolboxes as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import datetime
import warnings


# global setting
MAX_EPOCHS = 8
CHECK_STEPS = 1
N_PARTITIONS = 16
N_SEGMENTS = 1
LEARNING_RATE = 0.01
N_RECURRENT_STEP = 24
DATA_SHIFT = 16
IF_NODE_MODE = True
IF_IMAGE_LOG = True
IF_DATA_LOG = False
LOG_EXTRA_PREFIX = 'Estimation3_'



# load in data
current_dir = os.getcwd()
print('working directory is ' + current_dir)
if current_dir.split('/')[-1] == "dcm_rnn":
    os.chdir(current_dir + '/experiments/infer_x_from_y')
data_path = "../../dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)


# build model
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.learning_rate = LEARNING_RATE
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
neural_parameter_initial ={}
neural_parameter_initial['A'] = du.get('A')
neural_parameter_initial['B'] = du.get('B')
neural_parameter_initial['C'] = du.get('C')
dr.build_main_graph(neural_parameter_initial=neural_parameter_initial)
# dr.build_an_initializer_graph(hemodynamic_parameter_initial=None)
