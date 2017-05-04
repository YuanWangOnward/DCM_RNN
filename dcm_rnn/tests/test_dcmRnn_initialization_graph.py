from unittest import TestCase
from dcm_rnn import toolboxes as tb
from dcm_rnn.tf_model import DcmRnn
import tensorflow as tf
import numpy as np
import os

class TestDcmRnnInitializationGraph(TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.dr = DcmRnn()
        print(os.getcwd())
        data_path = 'dcm_rnn/resources/template0.pkl'
        self.du = tb.load_template(data_path)
        self.dr.collect_parameters(self.du)

    def test_initializer_graph_forward_pass(self):
        du = self.du
        dr = self.dr

        # build model
        dr.build_an_initializer_graph()
        isess = tf.InteractiveSession()
        isess.run(tf.global_variables_initializer())

        # prepare data
        # split_data_for_initializer_graph(x_data, y_data, n_segment, n_step, shift_x_h):
        data = {'x': tb.split_data_for_initializer_graph(
                    du.get('x'), du.get('y'), dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)[0],
                'y': tb.split_data_for_initializer_graph(
                    du.get('x'), du.get('y'), dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)[1],
                'h': tb.split(du.get('h'), dr.n_recurrent_step, 0)}

        h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region).astype(np.float32)

        # run forward
        y_predicted, h_state_predicted = dr.run_initializer_graph(isess, h_state_initial, data['x'])

        # merge results
        y_predicted = tb.merge(y_predicted, n_segment=dr.n_recurrent_step, n_step=dr.shift_data)

        # test
        np.testing.assert_array_almost_equal(
            np.array(du.get('y')[dr.shift_x_y: dr.shift_x_y + y_predicted.shape[0], :], dtype=np.float32),
            np.array(y_predicted, dtype=np.float32),
            decimal=4)



