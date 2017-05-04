from unittest import TestCase
import tensorflow as tf
from dcm_rnn import toolboxes as tb
from dcm_rnn.tf_model import DcmRnn
import numpy as np
import dcm_rnn.tf_model as tfm
import tensorflow as tf
import os
import random


class TestDcmRnnMainGraph(TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.dr = DcmRnn()
        print(os.getcwd())
        data_path = 'dcm_rnn/resources/template0.pkl'
        self.du = tb.load_template(data_path)
        self.dr.collect_parameters(self.du)
        self.neural_parameter_initial = {'A': self.du.get('A'), 'B': self.du.get('B'), 'C': self.du.get('C')}

    def test_build_main_graph(self):
        self.dr.build_main_graph(neural_parameter_initial=self.neural_parameter_initial)

    def test_main_graph_forward_pass(self):
        du = self.du
        dr = self.dr

        # prepare data
        # split_data_for_initializer_graph(x_data, y_data, n_segment, n_step, shift_x_h):
        data = {'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
                'x': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
                'h': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
                'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}

        # build model
        dr.build_main_graph(neural_parameter_initial=self.neural_parameter_initial)

        # run forward
        isess = tf.InteractiveSession()
        isess.run(tf.global_variables_initializer())

        for i in [random.randint(16, len(data['y'])) for _ in range(16)]:
            x_hat, h_hat, y_hat = isess.run([dr.x_monitor, dr.h_monitor, dr.y_predicted],
                                            feed_dict={
                                                dr.u_placeholder: data['u'][i],
                                                dr.x_state_initial: data['x'][i][0, :],
                                                dr.h_state_initial: data['h'][i][0, :, :],
                                            })
            np.testing.assert_array_almost_equal(
                np.array(data['x'][i], dtype=np.float32),
                np.array(x_hat, dtype=np.float32))

            np.testing.assert_array_almost_equal(
                np.array(data['h'][i], dtype=np.float32),
                np.array(h_hat, dtype=np.float32))

            # print(np.array(data['y'][i], dtype=np.float32))
            # print(np.array(y_hat, dtype=np.float32))
            np.testing.assert_array_almost_equal(
                np.array(data['y'][i], dtype=np.float32),
                np.array(y_hat, dtype=np.float32), decimal=4)

    def test_main_graph_loss(self):
        du = self.du
        dr = self.dr

        # prepare data
        # split_data_for_initializer_graph(x_data, y_data, n_segment, n_step, shift_x_h):
        data = {'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
                'x': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
                'h': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
                'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}

        # build model
        h_prior = self.dr.get_expanded_hemodynamic_parameter_prior_distributions(self.dr.n_region)
        mean = h_prior['mean'].astype(np.float32)
        std = h_prior['std'].astype(np.float32)
        h_parameters_initial = mean + std
        dr.build_main_graph(neural_parameter_initial=self.neural_parameter_initial,
                            hemodynamic_parameter_initial=h_parameters_initial)

        # run forward
        isess = tf.InteractiveSession()
        isess.run(tf.global_variables_initializer())

        for i in [random.randint(16, len(data['y'])) for _ in range(16)]:
            loss_prediction, loss_sparsity, loss_prior, loss_total = \
                isess.run([dr.loss_prediction, dr.loss_sparsity, dr.loss_prior, dr.loss_total],
                          feed_dict={
                              dr.u_placeholder: data['u'][i],
                              dr.x_state_initial: data['x'][i][0, :],
                              dr.h_state_initial: data['h'][i][0, :, :],
                              dr.y_true: data['y'][i]
                          })

            np.testing.assert_almost_equal(loss_sparsity, 0.3875, decimal=5)
            np.testing.assert_almost_equal(loss_prior, 15., decimal=5)
            np.testing.assert_almost_equal(loss_total, loss_prediction + loss_sparsity + loss_prior, decimal=5)

    def test_main_graph_loss_2(self):
        du = self.du
        dr = self.dr

        # prepare data
        # split_data_for_initializer_graph(x_data, y_data, n_segment, n_step, shift_x_h):
        data = {'u': tb.split(du.get('u'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data),
                'x': tb.split(du.get('x'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=0),
                'h': tb.split(du.get('h'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=1),
                'y': tb.split(du.get('y'), n_segment=dr.n_recurrent_step, n_step=dr.shift_data, shift=dr.shift_u_y)}

        # build model
        h_prior = self.dr.get_expanded_hemodynamic_parameter_prior_distributions(self.dr.n_region)
        mean = h_prior['mean'].astype(np.float32)
        h_parameters_initial = mean
        dr.build_main_graph(neural_parameter_initial=self.neural_parameter_initial,
                            hemodynamic_parameter_initial=h_parameters_initial)

        # run forward
        isess = tf.InteractiveSession()
        isess.run(tf.global_variables_initializer())

        for i in [random.randint(16, len(data['y'])) for _ in range(16)]:
            loss_prediction, loss_sparsity, loss_prior, loss_total = \
                isess.run([dr.loss_prediction, dr.loss_sparsity, dr.loss_prior, dr.loss_total],
                          feed_dict={
                              dr.u_placeholder: data['u'][i],
                              dr.x_state_initial: data['x'][i][0, :],
                              dr.h_state_initial: data['h'][i][0, :, :],
                              dr.y_true: data['y'][i]
                          })

            np.testing.assert_almost_equal(loss_prediction, 0, decimal=5)
            np.testing.assert_almost_equal(loss_sparsity, 0.3875, decimal=5)
            np.testing.assert_almost_equal(loss_prior, 0, decimal=5)
            np.testing.assert_almost_equal(loss_total, loss_prediction + loss_sparsity + loss_prior, decimal=5)
