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

# build model
dr = tfm.DcmRnn()
dr.collect_parameters(du)
# for initializer graph, hemodynamic parameters are set non-trainable
for key in dr.trainable_flags_h.keys():
    dr.trainable_flags_h[key] = False
dr.build_an_initializer_graph(hemodynamic_parameter_initial=None)


# prepare data
data = {}
data['y'] = tb.split(du.get('y'), dr.n_recurrent_step)
n_segments = len(data['y'])
data['x_hat'] = [np.zeros([dr.n_recurrent_step, dr.n_region]) for _ in range(n_segments)]

# training
# Launch the graph
TRAIN_EPOCHS = 1000
DISPLAY_STEP = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    for epoch in range(TRAIN_EPOCHS):
        for idx in range(n_segments):
            sess.run(dr.x_state_placeholder, feed_dict={dr.x_state_placeholder: data['x_hat'][idx]})
            _, data['x_hat'][idx] = sess.run([dr.train, dr.x_state], feed_dict={dr.y_true: data['y'][idx]})

        #Display logs per epoch step
        loss_total = 0
        if epoch % DISPLAY_STEP == 0:
            sess.run(dr.clear_loss_total)
            for idx in range(n_segments):
                sess.run(dr.x_state_placeholder, feed_dict={dr.x_state_placeholder: data['x_hat'][idx]})
                sess.run(dr.loss, feed_dict={dr.y_true: data['y'][idx]})
                loss_total = sess.run(dr.sum_loss)
            summary = sess.run(dr.merged_summary)
            dr.summary_writer.add_summary(summary, epoch)
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(loss_total))

    print("Optimization Finished!")


