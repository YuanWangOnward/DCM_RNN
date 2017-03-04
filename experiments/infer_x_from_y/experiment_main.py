import tensorflow as tf
import DCM_RNN.tf_model as tfm
import DCM_RNN.toolboxes as tb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import importlib

importlib.reload(tfm)
importlib.reload(tb)

# load in data
print('working directory is ' + os.getcwd())
data_path = "../../DCM_RNN/resources/template0.pkl"
du = tb.load_template(data_path)

dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.build_an_initializer_graph()

isess = tf.InteractiveSession()
tfm.reset_interactive_sesssion(isess)

opt_init_all = tf.global_variables_initializer()
isess.run(opt_init_all)

x_state = du.get('x')
y_state_predicted, h_state_predicted = isess.run([dr.y_state_predicted,
                                                  dr.h_state_predicted],
                                                 feed_dict={dr.x_state: x_state[:dr.n_recurrent_step, :]})

y_true = du.get('y')

np.testing.assert_array_equal(np.array(du.get('hemodynamic_parameter'), dtype=np.float32),
                              np.array(isess.run(dr.hemodynamic_parameter), dtype=np.float32))

np.testing.assert_array_almost_equal(np.array(du.get('h')[:dr.n_recurrent_step, :, :], dtype=np.float32),
                                     np.array(h_state_predicted, dtype=np.float32))

data = {'x': tb.split(du.get('x'), dr.n_recurrent_step)}

y_predicted = []
h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region).astype(np.float32)
for x_segment in data['x']:
    y_segment, h_state_final = isess.run([dr.y_state_predicted, dr.h_state_final],
                                         feed_dict={dr.x_state: x_segment, dr.h_state_initial: h_state_initial})
    h_state_initial = h_state_final
    y_predicted.append(y_segment)

y_predicted = np.concatenate(y_predicted)

plt.plot(y_predicted - y_true)
plt.plot(y_predicted)


# print(np.array(du.get('h')[:dr.n_recurrent_step, :, :], dtype=np.float32) -
#      np.array(h_state_predicted, dtype=np.float32))
# np.testing.assert_array_equal(np.array(y_state_predicted), y_true[:dr.n_recurrent_step, :])
# isess.run( , feed_dict=)
