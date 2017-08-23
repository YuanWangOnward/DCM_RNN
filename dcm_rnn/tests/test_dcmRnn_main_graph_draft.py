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
import warnings


class TestDcmRnnMainGraph(TestCase):
    MAX_EPOCHS = 1
    CHECK_STEPS = 1
    N_SEGMENTS = 64
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

    for k in data.keys():
        data[k] = data[k][: N_SEGMENTS]

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
        h_parameters_read_out = isess.run(dr.h_parameter_inital)

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

        STEP_SIZE = 0.1

        def apply_and_check(isess, grads_and_vars, step_size):
            isess.run([tf.assign(dr.Wxx, -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]),
                       tf.assign(dr.Wxxu[0], -grads_and_vars[1][0] * step_size + grads_and_vars[1][1]),
                       tf.assign(dr.Wxu, -grads_and_vars[2][0] * step_size + grads_and_vars[2][1])])
            loss_prediction = isess.run([dr.loss_prediction],
                                        feed_dict={
                                            dr.u_placeholder: data['u'][i],
                                            dr.x_state_initial: data['x_initial'][i][0, :],
                                            dr.h_state_initial: data['h_initial'][i][0, :, :],
                                            dr.y_true: data['y_true_float_corrected'][i]
                                        })
            return loss_prediction

        for r in range(1):
            for c in range(1):
                Wxx = du.get('Wxx')
                if Wxx[r, c] != 0.:
                    Wxx[r, c] = Wxx[r, c] * 0.9
                    isess.run(tf.assign(dr.Wxx, Wxx))

                    gradient_sum = 0
                    checking_loss = []
                    for epoch in range(self.MAX_EPOCHS):
                        for i in [random.randint(0, len(data['y_true_float_corrected']))
                                  for _ in range(self.N_TEST_SAMPLE)]:
                        # for i in range(len(SPM_data['y_true_float_corrected'])):

                            print('current processing ' + str(i))
                            grads_and_vars, loss_prediction = \
                                isess.run([dr.grads_and_vars, dr.loss_prediction],
                                          feed_dict={
                                              dr.u_placeholder: data['u'][i],
                                              dr.x_state_initial: data['x_initial'][i][0, :],
                                              dr.h_state_initial: data['h_initial'][i][0, :, :],
                                              dr.y_true: data['y_true_float_corrected'][i]
                                          })
                            gradient_sum += grads_and_vars[0][0]
                            # print('r=' + str(r) + ' c=' + str(c) + ' i=' + str(i) +
                            #       ' loss_prediction=' + str(loss_prediction))
                            # print(grads_and_vars[0][0])

                            # updating with back-tracking
                            step_size = STEP_SIZE
                            loss_prediction_original = loss_prediction
                            loss_prediction = apply_and_check(isess, grads_and_vars, step_size)

                            count = 0
                            while (loss_prediction > loss_prediction_original):
                                count += 1
                                if count == 16:
                                    step_size = 0
                                else:
                                    step_size = step_size / 2
                                print('step_size=' + str(step_size))
                                loss_prediction = apply_and_check(isess, grads_and_vars, step_size)

                            checking_loss.append(loss_prediction_original - loss_prediction)
                            print(checking_loss[-1])

                    print(gradient_sum)
                    print(grads_and_vars[0][1])
                    print(grads_and_vars[1][1])
                    print(grads_and_vars[2][1])
                    # print(loss_differences)

    def test_optimization(self):
        du = self.du
        dr = self.dr
        isess = self.isess
        data = self.data

        STEP_SIZE = 0.25

        def apply_and_check(isess, grads_and_vars, step_size, u, x_connector, h_connector, y_true):
            isess.run([tf.assign(dr.Wxx, -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]),
                       tf.assign(dr.Wxxu[0], -grads_and_vars[1][0] * step_size + grads_and_vars[1][1]),
                       tf.assign(dr.Wxu, -grads_and_vars[2][0] * step_size + grads_and_vars[2][1])])
            loss_prediction, x_connector, h_connector \
                = isess.run([dr.loss_prediction, dr.x_connector, dr.h_connector],
                            feed_dict={
                                dr.u_placeholder: u,
                                dr.x_state_initial: x_connector,
                                dr.h_state_initial: h_connector,
                                dr.y_true: y_true
                            })
            return [loss_prediction, x_connector, h_connector]

        def check_transition_matrix(Wxx):
            w, v = np.linalg.eig(Wxx)
            if max(w.real) < 1:
                return True
            else:
                return False

        for r in range(1):
            for c in range(1):
                Wxx = du.get('Wxx')
                if Wxx[r, c] != 0.:
                    Wxx[r, c] = Wxx[r, c] * 0.9
                    isess.run(tf.assign(dr.Wxx, Wxx))

                    gradient_sum = 0
                    checking_loss = []
                    for epoch in range(self.MAX_EPOCHS):
                        x_connector_current = dr.set_initial_neural_state_as_zeros(dr.n_region)
                        h_connector_current = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region)
                        for i in range(len(data['y_true_float_corrected'])):
                            print('current processing ' + str(i))
                            #print('u:')
                            #print(SPM_data['u'][i])
                            #print('x_initial:')
                            #print(x_connector_current)
                            #print('h_initial:')
                            #print(h_connector_current)
                            #print('y_true:')
                            #print(SPM_data['y_true_float_corrected'][i])


                            grads_and_vars, x_connector, h_connector, loss_prediction = \
                                isess.run([dr.grads_and_vars, dr.x_connector, dr.h_connector, dr.loss_prediction],
                                          feed_dict={
                                              dr.u_placeholder: data['u'][i],
                                              dr.x_state_initial: x_connector_current,
                                              dr.h_state_initial: h_connector_current,
                                              dr.y_true: data['y_true_float_corrected'][i]
                                          })
                            loss_prediction_original = loss_prediction
                            gradient_sum += grads_and_vars[0][0]
                            # print('r=' + str(r) + ' c=' + str(c) + ' i=' + str(i) +
                            #       ' loss_prediction=' + str(loss_prediction))
                            # for item in grads_and_vars:
                            #    print(item)


                            # updating with back-tracking
                            step_size = STEP_SIZE
                            loss_prediction, x_connector, h_connector = \
                                apply_and_check(isess, grads_and_vars, step_size, data['u'][i],
                                                              x_connector_current,
                                                              h_connector_current,
                                                              data['y_true_float_corrected'][i])

                            count = 0
                            while (loss_prediction > loss_prediction_original):
                                count += 1
                                if count == 16:
                                    step_size = 0
                                else:
                                    step_size = step_size / 2
                                print('step_size=' + str(step_size))
                                loss_prediction, x_connector, h_connector = \
                                    apply_and_check(isess, grads_and_vars,
                                                    step_size, data['u'][i], x_connector_current,
                                                    h_connector_current, data['y_true_float_corrected'][i])

                            Wxx = isess.run(dr.Wxx)
                            stable_flag = check_transition_matrix(Wxx)
                            while not stable_flag:
                                count += 1
                                if count == 16:
                                    step_size = 0
                                else:
                                    step_size = step_size / 2
                                warnings.warn('not stable')
                                print('step_size=' + str(step_size))
                                Wxx = -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]
                                stable_flag = check_transition_matrix(Wxx)
                            isess.run([tf.assign(dr.Wxx, -grads_and_vars[0][0] * step_size + grads_and_vars[0][1]),
                                       tf.assign(dr.Wxxu[0],
                                                 -grads_and_vars[1][0] * step_size + grads_and_vars[1][1]),
                                       tf.assign(dr.Wxu, -grads_and_vars[2][0] * step_size + grads_and_vars[2][1])])


                            x_connector_current = x_connector
                            h_connector_current = h_connector
                            checking_loss.append(loss_prediction_original - loss_prediction)
                            Wxxu, Wxu = isess.run([dr.Wxxu[0], dr.Wxu])

                            print(np.linalg.norm(data['y_true_float_corrected'][i].flatten()))
                            print(checking_loss[-1])
                            print(Wxx)
                            print(Wxxu)
                            #print(Wxx + Wxxu)
                            print(Wxu)


                    print('optimization finished.')
                    # print(gradient_sum)
                    print(grads_and_vars[0][1])
                    print(grads_and_vars[1][1])
                    print(grads_and_vars[2][1])
                    # print(loss_differences)
