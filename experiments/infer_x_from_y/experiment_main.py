import tensorflow as tf
import DCM_RNN.tf_model as tfm
import DCM_RNN.toolboxes as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

importlib.reload(tfm)
importlib.reload(tb)

# load in data
print('working directory is ' + os.getcwd())
if os.getcwd() == '/Users/yuanwang/Google_Drive/projects/Gits/DCM-RNN':
    os.chdir(os.getcwd() + '/experiments/infer_x_from_y')
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
data['y_true'] = tb.split(du.get('y'), dr.n_recurrent_step)
# n_segments = len(data['y_true'])
n_segments = 1
n_time_point_testing = n_segments * dr.n_recurrent_step
data['x_hat'] = [np.zeros([dr.n_recurrent_step, dr.n_region]) for _ in range(n_segments)]

# training
# Launch the graph

TRAIN_EPOCHS = 10
DISPLAY_STEP = 1
isess = tf.InteractiveSession()
sess = isess

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # show initial states

    epoch = 0
    sess.run(dr.clear_loss_total)
    for idx in range(n_segments):
        sess.run(dr.assign_x, feed_dict={dr.x_state_placeholder: data['x_hat'][idx]})
        loss_total = sess.run(dr.sum_loss, feed_dict={dr.y_true: data['y_true'][idx]})
    summary = sess.run(dr.merged_summary)
    dr.summary_writer.add_summary(summary, epoch)
    x_hat_temp = np.concatenate(data['x_hat'], axis=0)
    print("Epoch:", '%04d' % (epoch ), "y_total_loss=", "{:.9f}".format(loss_total))
    print("Epoch:", '%04d' % (epoch ),
          "x_total_loss=", "{:.9f}".format(tb.mse(x_hat_temp, du.get('x')[:n_time_point_testing, :])))
    plt.figure()
    plt.plot(x_hat_temp[:n_time_point_testing, :])
    plt.plot(du.get('x')[:n_time_point_testing, :])


    # Fit all training data
    h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region)
    for epoch in range(TRAIN_EPOCHS):
        for idx in range(n_segments):
            sess.run(dr.assign_x, feed_dict={dr.x_state_placeholder: data['x_hat'][idx]})
            sess.run(tf.assign(dr.h_state_initial, h_state_initial))

            _, data['x_hat'][idx], _ = sess.run(
                [dr.train, dr.x_state, dr.h_state_final],
                feed_dict={dr.y_true: data['y_true'][idx]})

        #Display logs per epoch step
        loss_total = 0
        h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region)
        if epoch % DISPLAY_STEP == 0:
            sess.run(dr.clear_loss_total)
            for idx in range(n_segments):
                sess.run(dr.assign_x, feed_dict={dr.x_state_placeholder: data['x_hat'][idx]})
                sess.run(tf.assign(dr.h_state_initial, h_state_initial))
                loss_total, h_state_initial = sess.run(
                    [dr.sum_loss, dr.h_state_final],
                    feed_dict={dr.y_true: data['y_true'][idx]})

            summary = sess.run(dr.merged_summary)
            dr.summary_writer.add_summary(summary, epoch + 1)
            x_hat_temp = np.concatenate(data['x_hat'], axis=0)
            print("Epoch:", '%04d' % (epoch+1), "y_total_loss=", "{:.9f}".format(loss_total))
            print("Epoch:", '%04d' % (epoch+1),
                  "x_total_loss=", "{:.9f}".format(tb.mse(x_hat_temp, du.get('x')[:n_time_point_testing, :])))
            plt.figure()
            plt.plot(x_hat_temp[:n_time_point_testing, :])
            plt.plot(du.get('x')[:n_time_point_testing, :])

    print("Optimization Finished!")

'''
plt.plot(x_hat_temp[:n_time_point_testing, :])
plt.plot(du.get('x')[:n_time_point_testing, :])
'''




