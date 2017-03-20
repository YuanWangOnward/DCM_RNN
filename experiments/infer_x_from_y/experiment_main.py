import tensorflow as tf
import DCM_RNN.tf_model as tfm
import DCM_RNN.toolboxes as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import copy

importlib.reload(tfm)
importlib.reload(tb)

# load in data
current_dir = os.getcwd()
print('working directory is ' + current_dir)
if current_dir.split('/')[-1] == "DCM-RNN":
    os.chdir(current_dir + '/experiments/infer_x_from_y')
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
data['x_true'], data['y_true'] = tb.split_data_for_initializer_graph(
    du.get('x'), du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift_x_y=dr.shift_x_y)

# n_segments = len(data['y_true'])
n_segments = 4
n_time_point_testing = n_segments * dr.n_recurrent_step
data['x_true'] = data['x_true'][:n_segments]
data['y_true'] = data['y_true'][:n_segments]
data['x_hat'] = [np.zeros([dr.n_recurrent_step, dr.n_region]) for _ in range(n_segments)]

# training
TRAIN_EPOCHS = 50
DISPLAY_STEP = 5
STOP_THRESHOLD = 1e-3
x_hat_previous = copy.deepcopy(data['x_hat'])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # show initial states
    epoch = 0
    sess.run(dr.clear_loss_total)
    h_initial_segment = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)
    for idx in range(n_segments):
        # assign proper values
        sess.run(dr.assign_x_state_stacked, feed_dict={dr.x_state_stacked_placeholder: data['x_hat'][idx]})
        sess.run(tf.assign(dr.h_state_initial, h_initial_segment))
        loss_total, h_initial_segment = sess.run(
            [dr.sum_loss, dr.h_connector],
            feed_dict={dr.y_true: data['y_true'][idx]})
    summary = sess.run(dr.merged_summary)
    dr.summary_writer.add_summary(summary, epoch)
    x_hat_temp = np.concatenate(data['x_hat'], axis=0)
    print("Epoch:", '%04d' % (epoch), "y_total_loss=", "{:.9f}".format(loss_total))
    print("Epoch:", '%04d' % (epoch),
          "x_total_loss=", "{:.9f}".format(tb.mse(x_hat_temp, du.get('x')[:n_time_point_testing, :])))
    plt.figure()
    plt.plot(x_hat_temp[:n_time_point_testing, :])
    plt.plot(du.get('x')[:n_time_point_testing, :])

    # Fit all training data
    for epoch in range(TRAIN_EPOCHS):
        h_initial_segment = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)
        for idx in range(n_segments):
            # assign proper data
            sess.run(dr.assign_x_state_stacked, feed_dict={dr.x_state_stacked_placeholder: data['x_hat'][idx]})
            sess.run(tf.assign(dr.h_state_initial, h_initial_segment))
            # training
            sess.run(dr.train, feed_dict={dr.y_true: data['y_true'][idx]})
            # collect results
            data['x_hat'][idx], h_initial_segment = sess.run([dr.x_state_stacked, dr.h_connector])

        # Display logs per epoch step
        # loss_total = 0
        h_initial_segment = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)
        if epoch % DISPLAY_STEP == 0:
            sess.run(dr.clear_loss_total)
            for idx in range(n_segments):
                # assign proper value
                sess.run(dr.assign_x_state_stacked, feed_dict={dr.x_state_stacked_placeholder: data['x_hat'][idx]})
                sess.run(tf.assign(dr.h_state_initial, h_initial_segment))
                # calculate loss and prepare for next segment
                loss_total, h_initial_segment = sess.run(
                    [dr.sum_loss, dr.h_connector],
                    feed_dict={dr.y_true: data['y_true'][idx]})
            summary = sess.run(dr.merged_summary)
            dr.summary_writer.add_summary(summary, epoch + 1)
            x_hat_temp = np.concatenate(data['x_hat'], axis=0)
            print("Epoch:", '%04d' % (epoch + 1), "y_total_loss=", "{:.9f}".format(loss_total))
            print("Epoch:", '%04d' % (epoch + 1),
                  "x_total_loss=", "{:.9f}".format(tb.mse(x_hat_temp, du.get('x')[:n_time_point_testing, :])))

        # check stop criterion
            relative_change = tb.rmse(np.concatenate(x_hat_previous), np.concatenate(data['x_hat']))
            if relative_change < STOP_THRESHOLD:
                print('Relative change: ' + str(relative_change))
                print('Stop criterion met, stop training')
                break
            else:
                x_hat_previous = copy.deepcopy(data['x_hat'])


    print("Optimization Finished!")
    plt.figure()
    #plt.plot(x_hat_temp[:n_time_point_testing, :])
    #plt.plot(du.get('x')[:n_time_point_testing, :])
    x_hat = tb.merge(data['x_hat'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    x_true = tb.merge(data['x_true'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    plt.plot(x_hat)
    plt.plot(x_true)




# show predicted fMRI signal
h_initial_segment = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_predicted = dr.run_initializer_graph(sess, h_initial_segment, data['x_hat'][: n_segments])[0]
    y_true = dr.run_initializer_graph(sess, h_initial_segment, data['x_true'][: n_segments])[0]

y_predicted = tb.merge(y_predicted, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
y_true = tb.merge(y_true, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)

print(tb.rmse(y_true, y_predicted))

plt.figure()
plt.plot(y_true - y_predicted)

plt.figure()
plt.plot(y_true)
plt.title('y_true')
plt.figure()
plt.plot(y_predicted)
plt.title('y_predicted')






