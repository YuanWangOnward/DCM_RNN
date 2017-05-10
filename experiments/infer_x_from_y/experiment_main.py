import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
#import dcm_rnn.tf_model as tfm
#import dcm_rnn.toolboxes as tb
import tf_model as tfm
import toolboxes as tb
import numpy as np
import os
import pickle
import datetime
import warnings
import sys
import random


'''
# choose matplotlib backend depending on interpreter used
if sys.executable == '/Users/yuanwang/anaconda/envs/tensorFlow/bin/python':
    print("It's a local run on Yuan's laptop. Matplotlib uses MacOSX backend")
else:
    print("It's NOT a local run on Yuan's laptop, Matplotlib uses AGG backend")
    matplotlib.use('agg')
'''


def prepare_data(max_segments=None, node_index=None):
    global data
    global sequence_length
    global H_STATE_INITIAL
    global IF_NOISED_Y
    data = {}

    if IF_NOISED_Y:
        pass
    else:
        data['x_true'], data['y_true'] = tb.split_data_for_initializer_graph(
            du.get('x'), du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift_x_y=dr.shift_x_y)

    data['h_true_monitor'] = tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    data['h_true_monitor'] = data['h_true_monitor'][:len(data['y_true'])]
    temp = tb.merge(data['x_true'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    sequence_length = dr.n_recurrent_step + (len(data['x_true']) - 1) * dr.shift_data
    data['x_hat'] = tb.ArrayWrapper(np.zeros(temp.shape) * np.mean(du.get('y')),
                                    n_segment=dr.n_recurrent_step, n_step=dr.shift_data)

    if max_segments is not None:
        if max_segments > len(data['x_true']):
            warnings.warn("max_segments is larger than the length of available data", UserWarning)
            max_segments = len(data['x_true'])
        data['x_true'] = data['x_true'][:max_segments]
        data['y_true'] = data['y_true'][:max_segments]
        data['h_true_monitor'] = data['h_true_monitor'][:max_segments]
        sequence_length = dr.n_recurrent_step + (len(data['x_true']) - 1) * dr.shift_data
        data['x_hat'].data = np.take(data['x_hat'].data, range(0, sequence_length), axis=0)

    if node_index is not None:
        data['x_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['x_true']]
        data['y_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['y_true']]
        data['h_true_monitor'] = [np.take(array, node_index, 1) for array in data['h_true_monitor']]
        # data['x_hat'].data = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['x_hat']]
        data['x_hat'] = tb.ArrayWrapper(np.take(data['x_hat'].data, node_index, axis=1),
                                        n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
        H_STATE_INITIAL = H_STATE_INITIAL[node_index].reshape(1, 4)
    return data


def get_log_prefix(extra_prefix=''):
    prefix = extra_prefix \
             + 'node' + str(dr.n_region) \
             + '_segment' + str(N_SEGMENTS) \
             + '_learningRate' + str(dr.learning_rate).replace('.', 'p') \
             + '_recurrentStep' + str(N_RECURRENT_STEP) \
             + '_dataShift' + str(DATA_SHIFT) \
             + '_iteration' + str(count_total)
    return prefix


def calculate_log_data():
    global x_hat_log
    global x_hat_log_merged
    global x_true_log
    global x_true_log_merged

    global h_hat_monitor_log
    global h_hat_connector_log
    global h_true_monitor_log
    global h_true_connector_log
    global h_original_monitor_log
    global h_original_connector_log
    global h_train_monitor_log
    global h_train_connector_log

    global y_hat_log
    global y_hat_log_merged
    global y_true_log
    global y_true_log_merged
    global y_original_log
    global y_original_log_merged

    # x_hat_log = data['x_hat']
    x_hat_log = tb.split(data['x_hat'].get(), n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    if IF_NODE_MODE:
        x_hat_log = [array.reshape(dr.n_recurrent_step, 1) for array in x_hat_log]
    x_true_log = data['x_true']

    sess.run(tf.global_variables_initializer())
    h_initial_segment = H_STATE_INITIAL
    y_hat_log, h_hat_monitor_log, h_hat_connector_log = \
        dr.run_initializer_graph(sess, h_initial_segment, x_hat_log)

    sess.run(tf.global_variables_initializer())
    h_initial_segment = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)
    y_true_log, h_true_monitor_log, h_true_connector_log = \
        dr.run_initializer_graph(sess, h_initial_segment, x_true_log)

    y_origianl_log = data['y_true']

    x_hat_log_merged = tb.merge(x_hat_log, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    x_true_log_merged = tb.merge(x_true_log, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)

    y_hat_log_merged = tb.merge(y_hat_log, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    y_true_log_merged = tb.merge(y_true_log, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    y_original_log_merged = tb.merge(y_origianl_log, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)


def add_image_log(image_log_dir='./image_logs/', extra_prefix=''):
    if not os.path.exists(image_log_dir):
        os.makedirs(image_log_dir)
    log_file_name_prefix = get_log_prefix(extra_prefix)

    x_hat = x_hat_log_merged
    x_true = x_true_log_merged

    y_hat = y_hat_log_merged
    y_true = y_true_log_merged
    y_original = y_original_log_merged

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x_true)
    plt.plot(x_hat, '--')
    plt.title('x true and hat Iteration = ' + str(count_total))

    plt.subplot(2, 2, 2)
    plt.plot(y_original)
    plt.plot(y_hat, '--')
    plt.title('y true and hat Iteration = ' + str(count_total))

    plt.subplot(2, 2, 3)
    plt.plot(y_original)
    plt.plot(y_true, '--')
    plt.title('y error by h_para and h_init')

    plt.subplot(2, 2, 4)
    plt.plot()
    plt.plot(y_total_loss)
    plt.plot(x_total_loss, '--')
    plt.title('y and x total loss_prediction')
    plt.tight_layout()

    plot_file_name = image_log_dir + log_file_name_prefix + '.png'
    plt.savefig(plot_file_name)
    plt.close()


def add_data_log(data_log_dir='./data_logs/', extra_prefix=''):
    if not os.path.exists(data_log_dir):
        os.makedirs(data_log_dir)

    log_file_name_prefix = get_log_prefix(extra_prefix)

    data_saved = {}
    data_saved['h_parameter_initial'] = H_PARA_INITIAL
    data_saved['h_state_initial'] = H_STATE_INITIAL
    data_saved['iteration'] = count_total
    data_saved['x_hat'] = x_hat_log_merged
    data_saved['x_true'] = x_true_log_merged
    data_saved['y_hat'] = y_hat_log_merged
    data_saved['y_true'] = y_true_log_merged

    file_name = data_log_dir + log_file_name_prefix + '.pkl'
    pickle.dump(data_saved, open(file_name, "wb"))


# global setting
IF_RANDOM_H_PARA = False
IF_RANDOM_H_STATE_INIT = True
IF_NOISED_Y = False
IF_NODE_MODE = True
IF_IMAGE_LOG = True
IF_DATA_LOG = True
if IF_NODE_MODE:
    NODE_INDEX = 2
    MAX_EPOCHS = 4
    MAX_EPOCHS_INNER = 4
    N_SEGMENTS = 128     #total amount of data segments
    CHECK_STEPS = N_SEGMENTS * MAX_EPOCHS_INNER
    N_RECURRENT_STEP = 64
    LEARNING_RATE = 128 / N_RECURRENT_STEP
    DATA_SHIFT = 4
    SNR = 2
    LOG_EXTRA_PREFIX = 'single_node_'
else:
    MAX_EPOCHS = 4
    MAX_EPOCHS_INNER = 4
    N_SEGMENTS = 64  # total amount of data segments
    CHECK_STEPS = N_SEGMENTS * MAX_EPOCHS_INNER
    N_RECURRENT_STEP = 64
    LEARNING_RATE = 128 / N_RECURRENT_STEP
    DATA_SHIFT = 4
    SNR = 2
    LOG_EXTRA_PREFIX = 'all_node_'


# load in data
current_dir = os.getcwd()
print('working directory is ' + current_dir)
if current_dir.split('/')[-1] == "dcm_rnn":
    sys.path.append(current_dir)
    # os.chdir(current_dir + '/..')
    data_path = "../resources/template0.pkl"
elif current_dir.split('/')[-1] == "DCM_RNN":
    sys.path.append('dcm_rnn')
    data_path = "dcm_rnn/resources/template0.pkl"
elif current_dir.split('/')[-1] == "infer_x_from_y":
    data_path = "../../dcm_rnn/resources/template0.pkl"
else:
    # on HPC
    data_path = "/home/yw1225/projects/DCM_RNN/dcm_rnn/resources/template0.pkl"
du = tb.load_template(data_path)
print('Loading data done.')

# build model
dr = tfm.DcmRnn()
dr.collect_parameters(du)
dr.learning_rate = LEARNING_RATE
dr.shift_data = DATA_SHIFT
dr.n_recurrent_step = N_RECURRENT_STEP
if IF_NODE_MODE:
    dr.n_region = 1
for key in dr.trainable_flags.keys():
    # in the initialization graph, the hemodynamic parameters are not trainable
    dr.trainable_flags[key] = False
if IF_RANDOM_H_PARA:
    H_PARA_INITIAL = \
        dr.randomly_generate_hemodynamic_parameters(dr.n_region, deviation_constraint=2).astype(np.float32)
else:
    H_PARA_INITIAL = dr.get_standard_hemodynamic_parameters(n_node=dr.n_region).astype(np.float32)
if IF_RANDOM_H_STATE_INIT:
    H_STATE_INITIAL = du.get('h')[random.randint(64, du.get('h').shape[0] - 64)].astype(np.float32)
else:
    H_STATE_INITIAL = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)

dr.build_an_initializer_graph(hemodynamic_parameter_initial=H_PARA_INITIAL)
print('Building tf model done.')

# prepare data
if IF_NODE_MODE:
    prepare_data(max_segments=N_SEGMENTS, node_index=NODE_INDEX)
else:
    prepare_data(max_segments=N_SEGMENTS)
print('Data preparation done.')

# training
print('Start training.')
x_hat_previous = data['x_hat'].data.copy()
x_total_loss = np.zeros([MAX_EPOCHS * MAX_EPOCHS_INNER * N_SEGMENTS])
y_total_loss = np.zeros([MAX_EPOCHS * MAX_EPOCHS_INNER * N_SEGMENTS])
count_total = 0
with tf.Session() as sess:
    for epoch in range(MAX_EPOCHS):
        # h_initial_segment = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)
        h_initial_segment = H_STATE_INITIAL

        for i_segment in range(N_SEGMENTS):
            sess.run(tf.global_variables_initializer())
            sess.run(dr.clear_loss_total)

            for epoch_inner in range(MAX_EPOCHS_INNER):
                # assign proper data
                if IF_NODE_MODE is True:
                    sess.run(dr.assign_x_state_stacked,
                             feed_dict={dr.x_state_stacked_placeholder:
                                        data['x_hat'].get(i_segment).reshape(dr.n_recurrent_step, 1)})
                else:
                    sess.run(dr.assign_x_state_stacked,
                             feed_dict={dr.x_state_stacked_placeholder: data['x_hat'].get(i_segment)})
                sess.run(tf.assign(dr.h_state_initial, h_initial_segment))

                # training
                sess.run(dr.train, feed_dict={dr.y_true: data['y_true'][i_segment]})

                # collect results
                temp = sess.run([dr.x_state_stacked])
                data['x_hat'].set(i_segment, temp)

                if epoch_inner == MAX_EPOCHS_INNER - 1:
                    # predict the coming x
                    '''
                    if i_segment < N_SEGMENTS - 1 and epoch == 0:
                        temp = data['x_hat'].get(i_segment)
                        predicted_value = np.mean(temp[-1])
                        data['x_hat'].set(i_segment + 1, predicted_value)
                        data['x_hat'].set(i_segment, temp)
                    '''
                    # update hemodynamic state initial
                    h_initial_segment = sess.run(dr.h_connector)
                count_total += 1

                # Display logs per CHECK_STEPS step
                if count_total % CHECK_STEPS == 0:
                    sess.run(dr.clear_loss_total)
                    # h_initial_segment_test = \
                    #     dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)
                    h_initial_segment_test = H_STATE_INITIAL
                    for j_segment in range(N_SEGMENTS):
                        # assign proper value
                        if IF_NODE_MODE is True:
                            sess.run(dr.assign_x_state_stacked,
                                     feed_dict={dr.x_state_stacked_placeholder:
                                                    data['x_hat'].get(j_segment).reshape(dr.n_recurrent_step, 1)})
                        else:
                            sess.run(dr.assign_x_state_stacked,
                                     feed_dict={dr.x_state_stacked_placeholder: data['x_hat'].get(j_segment)})
                        sess.run(tf.assign(dr.h_state_initial, h_initial_segment_test))

                        # calculate loss_prediction and prepare for next segment
                        loss_total, h_initial_segment_test = sess.run(
                            [dr.sum_loss, dr.h_connector],
                            feed_dict={dr.y_true: data['y_true'][j_segment]})

                    summary = sess.run(dr.merged_summary)
                    dr.summary_writer.add_summary(summary, count_total)
                    x_hat_temp = data['x_hat'].data
                    x_true_temp = tb.merge(data['x_true'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
                    y_total_loss[count_total - 1] = loss_total
                    x_total_loss[count_total - 1] = tb.mse(x_hat_temp, x_true_temp)
                    print("Total iteration:", '%04d' % count_total, "y_total_loss=", "{:.9f}".format(loss_total - 1))
                    print("Total iteration:", '%04d' % count_total, "x_total_loss=", "{:.9f}".format(x_total_loss[count_total - 1]))

                    if IF_IMAGE_LOG or IF_DATA_LOG:
                        calculate_log_data()

                    if IF_IMAGE_LOG:
                        add_image_log(extra_prefix=LOG_EXTRA_PREFIX)

                    if IF_DATA_LOG:
                        add_data_log(extra_prefix=LOG_EXTRA_PREFIX)
        '''
        # check stop criterion
        relative_change = tb.rmse(x_hat_previous, data['x_hat'].get())
        if relative_change < dr.stop_threshold:
            print('Relative change: ' + str(relative_change))
            print('Stop criterion met, stop training')
        else:
            # x_hat_previous = copy.deepcopy(data['x_hat'])
            x_hat_previous = data['x_hat'].get().copy()
        '''

print("Optimization Finished!")
'''
# x_hat = tb.merge(data['x_hat'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
x_hat = data['x_hat'].get()
x_true = tb.merge(data['x_true'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
plt.figure()
plt.plot(x_true)
plt.plot(x_hat, '--')
'''


# show predicted fMRI signal
'''
h_initial_segment = dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_hat = dr.run_initializer_graph(sess, h_initial_segment, data['x_hat'][: N_SEGMENTS])[0]
    y_true = dr.run_initializer_graph(sess, h_initial_segment, data['x_true'][: N_SEGMENTS])[0]

y_hat = tb.merge(y_hat, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
y_true = tb.merge(y_true, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(y_true)
plt.subplot(1, 2, 1)
plt.plot(y_hat, '--')
plt.title('y_true and y_hat')
plt.subplot(1, 2, 2)
plt.plot(y_true - y_hat)
plt.title('error, rmse=' + str(tb.rmse(y_true, y_hat)))
'''

# save result
'''
data_saved = {}
data_saved['MAX_EPOCHS'] = MAX_EPOCHS
data_saved['CHECK_STEPS'] = CHECK_STEPS
data_saved['stop_threshold'] = dr.stop_threshold
data_saved['n_recurrent_step'] = dr.n_recurrent_step
data_saved['n_region'] = dr.n_regiond
data_saved['learning_rate'] = dr.learning_rate
data_saved['shift_x_y'] = dr.shift_x_y
data_saved['shift_data'] = dr.shift_data
data_saved['epoch_inner'] = epoch_inner
data_saved['N_SEGMENTS'] = N_SEGMENTS


data_saved['x_hat'] = x_hat
data_saved['x_true'] = x_true
data_saved['y_hat'] = y_hat
data_saved['y_true'] = y_true

dt = datetime.datetime.now()
file_name = "%04d%02d%02d%02d%02d%02d.pkl" % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
pickle.dump(data_saved, open(file_name, "wb"))
'''
