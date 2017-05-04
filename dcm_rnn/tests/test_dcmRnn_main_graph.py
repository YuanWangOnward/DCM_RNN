from unittest import TestCase
import tensorflow as tf
from dcm_rnn import toolboxes as tb
from dcm_rnn.tf_model import DcmRnn
import numpy as np
import dcm_rnn.tf_model as tfm
import tensorflow as tf
import os


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

        i = 0
        x_hat, h_hat, y_hat = isess.run([dr.x_monitor, dr.h_monitor, dr.y_predicted],
                                        feed_dict={
                                            dr.u_placeholder: data['u'][i],
                                            dr.x_state_initial: data['x'][i][0, :],
                                            dr.h_state_initial: data['h'][i][0, :, :],
                                        })
        np.testing.assert_array_almost_equal(
            np.array(data['x'][i], dtype=np.float32),
            np.array(x_hat, dtype=np.float32))

'''
        h_state_monitor = []
        h_state_connector = []
        y_predicted = []
        h_state_initial_segment = h_state_initial
        for x_segment in data_x:
            h_state_connector.append(h_state_initial_segment)
            y_segment, h_segment, h_connector = \
                sess.run([self.y_predicted_stacked, self.h_state_monitor_stacked, self.h_connector],
                         feed_dict={self.x_state_stacked: x_segment, self.h_state_initial: h_state_initial_segment})
            h_state_initial_segment = h_connector
            h_state_monitor.append(h_segment)
            y_predicted.append(y_segment)
        return [y_predicted, h_state_monitor, h_state_connector]


        # run forward
        y_predicted, h_state_predicted = dr.run_initializer_graph(isess, h_state_initial, data['x'])

        # merge results
        y_predicted = tb.merge(y_predicted, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)

        # test
        
        np.testing.assert_array_almost_equal(
            np.array(du.get('h'), dtype=np.float32),
            np.array(h_state_predicted, dtype=np.float32))
        
        np.testing.assert_array_almost_equal(
            np.array(du.get('y')[dr.shift_x_y: dr.shift_x_y + y_predicted.shape[0], :], dtype=np.float32),
            np.array(y_predicted, dtype=np.float32),
            decimal=4)
'''



