from unittest import TestCase
import tensorflow as tf
from dcm_rnn import toolboxes as tb
from dcm_rnn.tf_model import DcmRnn
import numpy as np
import dcm_rnn.tf_model as tfm
import tensorflow as tf
import os
import random
import copy


class TestDcmRnnMainGraph(TestCase):
    MAX_EPOCHS = 1
    CHECK_STEPS = 1
    N_SEGMENTS = 128
    N_RECURRENT_STEP = 4
    LEARNING_RATE = 0.01 / N_RECURRENT_STEP
    DATA_SHIFT = 2
    N_TEST_SAMPLE_MAX = 32

    print(os.getcwd())
    data_path = '../resources/template0.pkl'
    du = tb.load_template(data_path)

    dr = DcmRnn()
    dr.collect_parameters(du)
    dr.learning_rate = LEARNING_RATE
    dr.shift_data = DATA_SHIFT
    dr.n_recurrent_step = N_RECURRENT_STEP
    neural_parameter_initial = {'A': du.get('A'), 'B': du.get('B'), 'C': du.get('C')}
    dr.loss_weighting = {'prediction': 1., 'sparsity': 0., 'prior': 0., 'Wxx': 0., 'Wxxu': 0., 'Wxu': 0.}
    dr.trainable_flags = {'Wxx': True,
                          'Wxxu': True,
                          'Wxu': True,
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

    data = {
        'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
        'x_initial': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
        'h_initial': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
        'x_whole': tb.split(du.get('x'), n_segment=dr.n_recurrent_step + dr.shift_u_x, n_step=dr.shift_data, shift=0),
        'h_whole': tb.split(du.get('h'), n_segment=dr.n_recurrent_step + dr.shift_x_y, n_step=dr.shift_data, shift=1),
        'h_predicted': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
        'y_predicted': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}
    N_TEST_SAMPLE = min(N_TEST_SAMPLE_MAX, len(data['y_predicted']))

    isess = tf.InteractiveSession()

    def setUp(self):
        self.isess.run(tf.global_variables_initializer())

    def test_output_layer(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data

        for i in [random.randint(0, len(data['y_predicted'])) for _ in range(self.N_TEST_SAMPLE)]:
            feed_key = {}
            for j in range(dr.n_recurrent_step):
                feed_key[dr.h_predicted[j]] = data['h_predicted'][i][j]
            y_predicted_hat = isess.run(dr.y_predicted, feed_dict=feed_key)

            du_temp = copy.deepcopy(du)
            parameter_package = du_temp.collect_parameter_for_y_scan()
            parameter_package['h'] = data['h_predicted'][i].astype(np.float32)
            y_du = du_temp.scan_y(parameter_package)

            np.testing.assert_array_almost_equal(
                np.array(np.squeeze(y_du), dtype=np.float32),
                np.array(np.squeeze(y_predicted_hat), dtype=np.float32))


    def test_main_graph_forward_pass(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data

        for i in [random.randint(16, len(data['y_predicted'])) for _ in range(16)]:
            x_whole_hat, h_whole_hat, y_prediction_hat = isess.run([dr.x_whole, dr.h_whole, dr.y_predicted],
                                                        feed_dict={
                                                            dr.u_placeholder: data['u'][i],
                                                            dr.x_state_initial: data['x_initial'][i][0, :],
                                                            dr.h_state_initial: data['h_initial'][i][0, :, :],
                                                        })
            # u -> x
            np.testing.assert_array_almost_equal(
                np.array(data['x_whole'][i][:dr.n_recurrent_step + dr.shift_u_x, :], dtype=np.float32),
                np.array(x_whole_hat[:dr.n_recurrent_step + dr.shift_u_x], dtype=np.float32))

            # x -> h
            for j in range(dr.n_recurrent_step + dr.shift_x_y):
                n_temp = max(0, j - (dr.n_recurrent_step))
                np.testing.assert_array_almost_equal(
                    np.array(np.squeeze(data['h_whole'][i][j, :, n_temp:]), dtype=np.float32),
                    np.array(np.squeeze(h_whole_hat[j][:, n_temp:]), dtype=np.float32))
