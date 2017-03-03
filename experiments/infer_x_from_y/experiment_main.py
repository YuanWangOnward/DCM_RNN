import tensorflow as tf
import DCM_RNN.tf_model as tfm
import DCM_RNN.toolboxes as tb
import os

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

opt_init_all=tf.initialize_all_variables()
isess.run(opt_init_all)

x_state = du.get('x')
x_state =

# isess.run( , feed_dict=)