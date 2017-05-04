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

du = self.du
dr = self.dr

# prepare data
# split_data_for_initializer_graph(x_data, y_data, n_segment, n_step, shift_x_h):
data = {'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
        'x': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
        'h': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
        'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}

# build model
h_prior = self.dr.get_expanded_hemodynamic_parameter_prior_distributions(self.dr.n_region)
mean = h_prior['mean'].astype(np.float32)
std = h_prior['std'].astype(np.float32)
h_parameters_initial = mean + std
dr.build_main_graph(neural_parameter_initial=self.neural_parameter_initial,
                    hemodynamic_parameter_initial=h_parameters_initial)

# run forward
isess = tf.InteractiveSession()
isess.run(tf.global_variables_initializer())

for i in [random.randint(16, len(data['y'])) for _ in range(16)]:
    loss_prediction, loss_sparsity, loss_prior, loss_total = \
        isess.run([dr.loss_prediction, dr.loss_sparsity, dr.loss_prior, dr.loss_total],
                  feed_dict={
                      dr.u_placeholder: data['u'][i],
                      dr.x_state_initial: data['x'][i][0, :],
                      dr.h_state_initial: data['h'][i][0, :, :],
                      dr.y_true: data['y'][i]
                  })
