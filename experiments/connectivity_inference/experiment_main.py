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
MAX_EPOCHS = 2
CHECK_STEPS = 1
N_PARTITIONS = 16
N_SEGMENTS = 128
LEARNING_RATE = 0.00001
N_RECURRENT_STEP = 8
DATA_SHIFT = 4
IF_NODE_MODE = True
IF_IMAGE_LOG = True
IF_DATA_LOG = False
LOG_EXTRA_PREFIX = 'Estimation3_'

# load in data
current_dir = os.getcwd()
print('working directory is ' + current_dir)
if current_dir.split('/')[-1] == "dcm_rnn":
    os.chdir(current_dir + '/..')
    data_path = "/resources/template0.pkl"
elif current_dir.split('/')[-1] == "DCM_RNN":
    data_path = "/dcm_rnn/resources/template0.pkl"
elif current_dir.split('/')[-1] == "connectivity_inference":
    data_path = "../../dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)

# build model
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.learning_rate = LEARNING_RATE
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
neural_parameter_initial = {}
neural_parameter_initial['A'] = du.get('A') * 1.5
neural_parameter_initial['B'] = du.get('B')
neural_parameter_initial['C'] = du.get('C')
dr.loss_weighting = {'prediction': 5., 'sparsity': 1., 'prior': 1., 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
dr.trainable_flags = {'Wxx': True,
                      'Wxxu': False,
                      'Wxu': False,
                      'alpha': False,
                      'E0': False,
                      'k': False,
                      'gamma': False,
                      'tao': False,
                      'epsilon': False,
                      'V0': False,
                      'TE': False,
                      'r0': False,
                      'theta0': False,
                      'x_h_coupling': False
                      }
dr.build_main_graph(neural_parameter_initial=neural_parameter_initial)

# prepare data
data = {'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
        'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}
n_segment = min([len(data[x]) for x in data.keys()])
for k in data.keys():
    data[k] = data[k][: min([n_segment, N_SEGMENTS])]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_total_accumulated_list = []
    loss_prediction_accumulated_list = []
    for epoch in range(MAX_EPOCHS):
        x_state_initial = dr.set_initial_neural_state_as_zeros(dr.n_region).astype(np.float32)
        h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region).astype(np.float32)
        loss_total_accumulated = 0
        loss_prediction_accumulated = 0
        for i in range(len(data['u'])):
            _, loss_total, loss_prediction, x_state_initial, h_state_initial = \
                sess.run([dr.train, dr.loss_total, dr.loss_prediction, dr.x_connector, dr.h_connector],
                         feed_dict={
                             dr.u_placeholder: data['u'][i],
                             dr.x_state_initial: x_state_initial,
                             dr.h_state_initial: h_state_initial,
                             dr.y_true: data['y'][i]
                         })
            print("Index:", '%04d' % i, "loss_prediction=", "{:.9f}".format(loss_prediction))
            Wxx = sess.run(dr.Wxx)
            print(Wxx)
            loss_total_accumulated += loss_total
            loss_prediction_accumulated += loss_prediction
        loss_total_accumulated_list.append(loss_total_accumulated)
        loss_prediction_accumulated_list.append(loss_prediction_accumulated)
        if epoch % CHECK_STEPS == 0:
            print("Epoch:", '%04d' % epoch, "loss_total=", "{:.9f}".format(loss_total_accumulated),
                  "loss_prediction=", "{:.9f}".format(loss_prediction_accumulated))
            Wxx = sess.run(dr.Wxx)
            print(Wxx)
    Wxx = sess.run(dr.Wxx)
    Wxxu = sess.run(dr.Wxxu)
    Wxu = sess.run(dr.Wxu)

print("Optimization Finished!")
print(loss_total_accumulated_list)
print(loss_prediction_accumulated_list)
print(Wxx)
print(Wxxu[0])
print(Wxu)
