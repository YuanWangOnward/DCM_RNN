import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
# import dcm_rnn.tf_model as tfm
# import dcm_rnn.toolboxes as tb
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


def get_log_prefix(extra_prefix=''):
    prefix = extra_prefix \
             + 'nodeNumber' + str(dr.n_region) \
             + '_segments' + str(N_SEGMENTS) \
             + '_learningRate' + str(dr.learning_rate).replace('.', 'p') \
             + '_recurrentStep' + str(N_RECURRENT_STEP) \
             + '_dataShift' + str(DATA_SHIFT) \
             + '_iteration' + str(count_total)
    return prefix


def prepare_data(max_segments=None, node_index=None):
    global data
    global SEQUENCE_LENGTH
    global H_STATE_INITIAL
    global IF_NOISED_Y
    global SNR
    global NOISE

    if IF_NOISED_Y:
        std = np.std(du.get('y').reshape([-1])) / SNR
        NOISE = np.random.normal(0, std, du.get('y').shape)
    else:
        NOISE = np.zeros(du.get('y').shape)

    ## saved line. SEQUENCE_LENGTH = dr.n_recurrent_step + (len(data['x_true']) - 1) * dr.shift_data
    data['y_train'] = tb.split(du.get('y') + NOISE, dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)
    max_segments_natural = len(data['y_train'])
    data['y_true'] = tb.split(du.get('y'), dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)[:max_segments_natural]
    data['h_true_monitor'] = tb.split(du.get('h'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]
    data['x_true'] = tb.split(du.get('x'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]

    if max_segments is not None:
        if max_segments > max_segments_natural:
            warnings.warn("max_segments is larger than the length of available data", UserWarning)
        else:
            data['x_true'] = data['x_true'][:max_segments]
            data['h_true_monitor'] = data['h_true_monitor'][:max_segments]
            data['y_true'] = data['y_true'][:max_segments]
            data['y_train'] = data['y_train'][:max_segments]

    if node_index is not None:
        data['x_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['x_true']]
        data['h_true_monitor'] = [np.take(array, node_index, 1) for array in data['h_true_monitor']]
        data['y_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['y_true']]
        data['y_train'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['y_train']]
        H_STATE_INITIAL = H_STATE_INITIAL[node_index].reshape(1, 4)

    # collect merged data (without split and merge, it can be tricky to cut proper part from du)
    data['x_true_merged'] = tb.merge(data['x_true'], dr.n_recurrent_step, dr.shift_data)
    # x_hat is with extra wrapper for easy modification with a single index
    data['x_hat_merged'] = tb.ArrayWrapper(np.zeros(data['x_true_merged'].shape), dr.n_recurrent_step, dr.shift_data)
    data['h_true_monitor_merged'] = tb.merge(data['h_true_monitor'], dr.n_recurrent_step, dr.shift_data)
    data['y_true_merged'] = tb.merge(data['y_true'], dr.n_recurrent_step, dr.shift_data)
    data['y_train_merged'] = tb.merge(data['y_train'], dr.n_recurrent_step, dr.shift_data)

    return data


def calculate_log_data():

    # run forward pass with x_true to show y error caused by error in the network parameters, it should run only once
    if 'y_hat_x_true' not in data.keys():
        sess.run(tf.global_variables_initializer())
        y_hat_x_true_log, h_hat_x_true_monitor_log, h_hat_x_true_connector_log = \
            dr.run_initializer_graph(sess, H_STATE_INITIAL, data['x_true'])
        data['h_hat_x_true_monitor'] = h_hat_x_true_monitor_log
        data['y_hat_x_true'] = y_hat_x_true_log
        data['h_hat_x_true_monitor_merged'] = tb.merge(h_hat_x_true_monitor_log, dr.n_recurrent_step, dr.shift_data)
        data['y_hat_x_true_merged'] = tb.merge(y_hat_x_true_log, dr.n_recurrent_step, dr.shift_data)

    data['x_hat'] = tb.split(data['x_hat_merged'].get(), n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
    if IF_NODE_MODE:
        data['x_hat'] = [array.reshape(dr.n_recurrent_step, 1) for array in data['x_hat']]

    sess.run(tf.global_variables_initializer())
    y_hat_log, h_hat_monitor_log, h_hat_connector_log = \
        dr.run_initializer_graph(sess, H_STATE_INITIAL, data['x_hat'])

    # collect results
    # segmented data
    data['x_hat'] = data['x_hat']
    data['x_true'] = data['x_true']

    data['h_true_monitor'] = data['h_true_monitor']
    data['h_hat_x_true_monitor'] = data['h_hat_x_true_monitor']
    data['h_hat_monitor'] = h_hat_monitor_log

    data['y_train'] = data['y_train']
    data['y_true'] = data['y_true']
    data['y_hat_x_true'] = data['y_hat_x_true']
    data['y_hat'] = y_hat_log


    # merged data
    data['x_true_merged'] = data['x_true_merged']
    data['x_hat_merged'] = data['x_hat_merged']

    data['h_true_monitor_merged'] = data['h_true_monitor_merged']
    data['h_hat_x_true_monitor_merged'] = data['h_hat_x_true_monitor_merged']
    data['h_hat_monitor_merged'] = tb.merge(h_hat_monitor_log, dr.n_recurrent_step, dr.shift_data)


    data['y_train_merged'] = data['y_train_merged']
    data['y_true_merged'] = data['y_true_merged']
    data['y_hat_x_true_merged'] = data['y_hat_x_true_merged']
    data['y_hat_merged'] = tb.merge(y_hat_log, dr.n_recurrent_step, dr.shift_data)


def add_image_log(image_log_dir='./image_logs/', extra_prefix=''):
    global IF_NOISED_Y
    if not os.path.exists(image_log_dir):
        os.makedirs(image_log_dir)
    log_file_name_prefix = get_log_prefix(extra_prefix)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(data['x_true_merged'], label='x_true')
    plt.plot(data['x_hat_merged'].get(), '--', label='x_hat')
    plt.title('Iteration = ' + str(count_total))
    plt.legend()

    if IF_NOISED_Y:
        plt.subplot(2, 2, 2)
        plt.plot(data['y_train_merged'], label='y_train')
        plt.plot(data['y_true_merged'], label='y_true')
        plt.plot(data['y_hat_merged'], '--', label='y_hat')
        plt.title('Iteration = ' + str(count_total))
        plt.legend()
    else:
        plt.subplot(2, 2, 2)
        plt.plot(data['y_true_merged'], label='y_true')
        plt.plot(data['y_hat_merged'], '--', label='y_hat')
        plt.title('Iteration = ' + str(count_total))
        plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(data['y_true_merged'], label='by net_true')
    plt.plot(data['y_hat_x_true_merged'], '--', label='by net_hat')
    plt.title('y reproduced with x_true, iteration = ' + str(count_total))
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(y_total_loss, label='y loss')
    plt.plot(x_total_loss, '--', label='x loss')
    plt.title('Iteration = ' + str(count_total))
    plt.legend()

    plt.tight_layout()
    plot_file_name = image_log_dir + log_file_name_prefix + '.png'
    plt.savefig(plot_file_name)
    plt.close()


def add_data_log(data_log_dir='./data_logs/', extra_prefix=''):
    if not os.path.exists(data_log_dir):
        os.makedirs(data_log_dir)

    log_file_name_prefix = get_log_prefix(extra_prefix)

    data_saved = {}

    data_saved['IF_RANDOM_H_PARA'] = IF_RANDOM_H_PARA
    data_saved['IF_RANDOM_H_STATE_INIT'] = IF_RANDOM_H_STATE_INIT
    data_saved['IF_NOISED_Y'] = IF_NOISED_Y
    data_saved['IF_NODE_MODE'] = IF_NODE_MODE
    data_saved['IF_IMAGE_LOG'] = IF_IMAGE_LOG
    data_saved['IF_DATA_LOG'] = IF_DATA_LOG

    data_saved['NODE_INDEX'] = NODE_INDEX
    data_saved['MAX_EPOCHS'] = MAX_EPOCHS
    data_saved['MAX_EPOCHS_INNER'] = MAX_EPOCHS_INNER
    data_saved['N_SEGMENTS'] = N_SEGMENTS
    data_saved['CHECK_STEPS'] = CHECK_STEPS

    data_saved['N_RECURRENT_STEP'] = N_RECURRENT_STEP
    data_saved['LEARNING_RATE'] = LEARNING_RATE
    data_saved['DATA_SHIFT'] = DATA_SHIFT
    data_saved['SNR'] = SNR
    data_saved['LOG_EXTRA_PREFIX'] = LOG_EXTRA_PREFIX

    data_saved['iteration'] = count_total
    data_saved.update(data)

    file_name = data_log_dir + log_file_name_prefix + '.pkl'
    pickle.dump(data_saved, open(file_name, "wb"))


# global setting
IF_RANDOM_H_PARA = True
IF_RANDOM_H_STATE_INIT = True
IF_NOISED_Y = True

IF_NODE_MODE = True
IF_IMAGE_LOG = True
IF_DATA_LOG = True
if IF_NODE_MODE:
    NODE_INDEX = 0
    MAX_EPOCHS = 4
    MAX_EPOCHS_INNER = 4
    N_SEGMENTS = 128  # total amount of data segments
    CHECK_STEPS = N_SEGMENTS * MAX_EPOCHS_INNER
    N_RECURRENT_STEP = 64
    LEARNING_RATE = 128 / N_RECURRENT_STEP
    DATA_SHIFT = 4
    SNR = 3
    LOG_EXTRA_PREFIX = 'single_node_'
else:
    MAX_EPOCHS = 4
    MAX_EPOCHS_INNER = 4
    N_SEGMENTS = 64  # total amount of data segments
    CHECK_STEPS = N_SEGMENTS * MAX_EPOCHS_INNER
    N_RECURRENT_STEP = 64
    LEARNING_RATE = 128 / N_RECURRENT_STEP
    DATA_SHIFT = 4
    SNR = 3
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
data = {}
if IF_NODE_MODE:
    prepare_data(max_segments=N_SEGMENTS, node_index=NODE_INDEX)
else:
    prepare_data(max_segments=N_SEGMENTS)
print('Data preparation done.')

# training
print('Start training.')
x_hat_previous = data['x_hat_merged'].data.copy()
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
                                            data['x_hat_merged'].get(i_segment).reshape(dr.n_recurrent_step, 1)})
                else:
                    sess.run(dr.assign_x_state_stacked,
                             feed_dict={dr.x_state_stacked_placeholder: data['x_hat_merged'].get(i_segment)})
                sess.run(tf.assign(dr.h_state_initial, h_initial_segment))

                # training
                sess.run(dr.train, feed_dict={dr.y_true: data['y_train'][i_segment]})

                # collect results
                temp = sess.run([dr.x_state_stacked])
                data['x_hat_merged'].set(i_segment, temp)

                if epoch_inner == MAX_EPOCHS_INNER - 1:
                    # predict the coming x
                    '''
                    if i_segment < N_SEGMENTS - 1 and epoch == 0:
                        temp = data['x_hat_merged'].get(i_segment)
                        predicted_value = np.mean(temp[-1])
                        data['x_hat_merged'].set(i_segment + 1, predicted_value)
                        data['x_hat_merged'].set(i_segment, temp)
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
                                                    data['x_hat_merged'].get(j_segment).reshape(dr.n_recurrent_step,
                                                                                                1)})
                        else:
                            sess.run(dr.assign_x_state_stacked,
                                     feed_dict={dr.x_state_stacked_placeholder: data['x_hat_merged'].get(j_segment)})
                        sess.run(tf.assign(dr.h_state_initial, h_initial_segment_test))

                        # calculate loss_prediction and prepare for next segment
                        loss_total, h_initial_segment_test = sess.run(
                            [dr.sum_loss, dr.h_connector],
                            feed_dict={dr.y_true: data['y_train'][j_segment]})

                    summary = sess.run(dr.merged_summary)
                    dr.summary_writer.add_summary(summary, count_total)
                    x_hat_temp = data['x_hat_merged'].data
                    x_true_temp = data['x_true_merged']
                    y_total_loss[count_total - 1] = loss_total
                    x_total_loss[count_total - 1] = tb.mse(x_hat_temp, x_true_temp)
                    print("Total iteration:", '%04d' % count_total, "y_total_loss=", "{:.9f}".format(loss_total - 1))
                    print("Total iteration:", '%04d' % count_total, "x_total_loss=",
                          "{:.9f}".format(x_total_loss[count_total - 1]))

                    if IF_IMAGE_LOG or IF_DATA_LOG:
                        calculate_log_data()

                    if IF_IMAGE_LOG:
                        add_image_log(extra_prefix=LOG_EXTRA_PREFIX)

                    if IF_DATA_LOG:
                        add_data_log(extra_prefix=LOG_EXTRA_PREFIX)
        '''
        # check stop criterion
        relative_change = tb.rmse(x_hat_previous, data['x_hat_merged'].get())
        if relative_change < dr.stop_threshold:
            print('Relative change: ' + str(relative_change))
            print('Stop criterion met, stop training')
        else:
            # x_hat_previous = copy.deepcopy(data['x_hat_merged'])
            x_hat_previous = data['x_hat_merged'].get().copy()
        '''

print("Optimization Finished!")
'''
# x_hat = tb.merge(data['x_hat_merged'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
x_hat = data['x_hat_merged'].get()
x_true = tb.merge(data['x_true'], n_segment=dr.n_recurrent_step, n_step=dr.shift_data)
plt.figure()
plt.plot(x_true)
plt.plot(x_hat, '--')
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


data_saved['x_hat_merged'] = x_hat
data_saved['x_true'] = x_true
data_saved['y_hat'] = y_hat
data_saved['y_train'] = y_train

dt = datetime.datetime.now()
file_name = "%04d%02d%02d%02d%02d%02d.pkl" % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
pickle.dump(data_saved, open(file_name, "wb"))
'''
