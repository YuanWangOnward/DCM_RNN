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
    N_RECURRENT_STEP = 8
    LEARNING_RATE = 0.01 / N_RECURRENT_STEP
    DATA_SHIFT = 4
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
    dr.loss_weighting = {'prediction': 1., 'sparsity': 0., 'prior': 0., 'Wxx': 1., 'Wxxu': 1., 'Wxu': 1.}
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
        'y_true': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y),
        'y_true_float_corrected': []}

    for i in range(len(data['y_true'])):
        parameter_package = du.collect_parameter_for_y_scan()
        parameter_package['h'] = data['h_predicted'][i].astype(np.float32)
        data['y_true_float_corrected'].append(du.scan_y(parameter_package))

    N_TEST_SAMPLE = min(N_TEST_SAMPLE_MAX, len(data['y_true_float_corrected']))

    isess = tf.InteractiveSession()

    def setUp(self):
        self.isess.run(tf.global_variables_initializer())

    def test_output_layer(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data

        for i in range(len(data['y_true_float_corrected'])):
            feed_key = {}
            for j in range(dr.n_recurrent_step):
                feed_key[dr.h_predicted[j]] = data['h_predicted'][i][j]
            y_predicted_hat = isess.run(dr.y_predicted, feed_dict=feed_key)

            np.testing.assert_array_almost_equal(
                np.array(np.squeeze(data['y_true_float_corrected'][i]), dtype=np.float32),
                np.array(np.squeeze(y_predicted_hat), dtype=np.float32))

    def test_main_graph_forward_pass(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data

        for i in [random.randint(0, len(data['y_true_float_corrected'])) for _ in range(self.N_TEST_SAMPLE)]:
            x_whole_hat, h_whole_hat, y_predicted_hat = isess.run([dr.x_whole, dr.h_whole, dr.y_predicted],
                                                                  feed_dict={
                                                                      dr.u_placeholder: data['u'][i],
                                                                      dr.x_state_initial: data['x_initial'][i][0, :],
                                                                      dr.h_state_initial: data['h_initial'][i][0, :,
                                                                                          :],
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

    def test_update_h_parameters_in_graph(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data
        # TODO
        '''
        h_prior = dr.get_expanded_hemodynamic_parameter_prior_distributions(dr.n_region)
        mean = h_prior['mean'].astype(np.float32)
        std = h_prior['std'].astype(np.float32)
        h_parameters_updated = mean + std

        dr.update_h_parameters_in_graph(isess, h_parameters_updated)
        h_parameters_read_out = isess.run(dr.h_parameters)

        np.testing.assert_array_equal(h_parameters_updated, h_parameters_read_out)
        '''

    def test_main_graph_loss(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data

        for i in range(len(data['y_true_float_corrected'])):
            try:
                feed_dict = {
                    dr.u_placeholder: data['u'][i],
                    dr.x_state_initial: data['x_initial'][i][0, :],
                    dr.h_state_initial: data['h_initial'][i][0, :, :],
                    dr.y_true: np.zeros(data['y_true_float_corrected'][i].shape)
                }

                loss_prediction, loss_sparsity, loss_prior, loss_total, y_hat = \
                    isess.run([dr.loss_prediction, dr.loss_sparsity, dr.loss_prior, dr.loss_total, dr.y_predicted],
                              feed_dict=feed_dict)
                np.testing.assert_almost_equal(loss_prediction, tb.mse(np.array(y_hat, dtype=np.float32)), decimal=5)
                np.testing.assert_almost_equal(loss_sparsity, 0.019642857302512442)
                np.testing.assert_almost_equal(loss_prior, 0., decimal=5)
                loss_true = dr.loss_weighting['prediction'] * loss_prediction \
                            + dr.loss_weighting['sparsity'] * loss_sparsity \
                            + dr.loss_weighting['prior'] * loss_prior
                np.testing.assert_almost_equal(loss_total, loss_true, decimal=5)
            except Exception:
                print('i= ' + str(i))
                print('u:')
                print(data['u'][i])
                print('x_state_initial: ')
                print(data['x_initial'][i][0, :])
                print('h_state_initial: ')
                print(data['h_initial'][i][0, :, :])
                print('y_hat: ')
                print(y_hat)
                print('y_true: ')
                print(data['y_true_float_corrected'][i])
                raise Exception

        h_prior = self.dr.get_expanded_hemodynamic_parameter_prior_distributions(self.dr.n_region)
        mean = h_prior['mean'].astype(np.float32)
        std = h_prior['std'].astype(np.float32)
        h_parameters_updated = mean + std

        for i in [random.randint(0, len(data['y_true_float_corrected'])) for _ in range(self.N_TEST_SAMPLE)]:
            loss_prediction, loss_sparsity, loss_prior, loss_total = \
                isess.run([dr.loss_prediction, dr.loss_sparsity, dr.loss_prior, dr.loss_total],
                          feed_dict={
                              dr.u_placeholder: data['u'][i],
                              dr.x_state_initial: data['x_initial'][i][0, :],
                              dr.h_state_initial: data['h_initial'][i][0, :, :],
                              dr.y_true: np.zeros(data['y_true_float_corrected'][i].shape),
                              dr.h_parameters: h_parameters_updated
                          })
            np.testing.assert_almost_equal(loss_sparsity, 0.019642857302512442)
            np.testing.assert_almost_equal(loss_prior, 1., decimal=5)

    def test_gradient(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data

        for r in range(1):
            for c in range(1):
                Wxx = du.get('Wxx')
                if Wxx[r, c] != 0.:
                    Wxx[r, c] = Wxx[r, c] * 1.1
                    isess.run(tf.assign(dr.Wxx, Wxx))

                    gradient_sum = 0
                    for i in range(len(data['y_true_float_corrected'])):
                        grads_and_vars, loss_prediction = \
                            isess.run([dr.grads_and_vars, dr.loss_prediction],
                                      feed_dict={
                                          dr.u_placeholder: data['u'][i],
                                          dr.x_state_initial: data['x_initial'][i][0, :],
                                          dr.h_state_initial: data['h_initial'][i][0, :,
                                                              :],
                                          dr.y_true: data['y_true_float_corrected'][i]
                                      })
                        gradient_sum += grads_and_vars[0][0]
                        # print('r=' + str(r) + ' c=' + str(c) + ' i=' + str(i) +
                        #       ' loss_prediction=' + str(loss_prediction))
                        # print(grads_and_vars[0][0])

                        # updating with back-tracking


                    print(gradient_sum)
