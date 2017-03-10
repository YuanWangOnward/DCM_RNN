from unittest import TestCase
import tensorflow as tf
from DCM_RNN import toolboxes as tb
from DCM_RNN.tf_model import DcmRnn
import numpy as np
import DCM_RNN.tf_model as tfm
import tensorflow as tf
import numpy as np


class TestDcmRnn(TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.dr = DcmRnn()
        data_path = '../resources/template0.pkl'
        self.du = tb.load_template(data_path)
        self.dr.collect_parameters(self.du)

    def test_collect_parameters(self):
        parameter_package = self.dr.collect_parameters(self.du)

    def test_load_parameters(self):
        parameter_package = self.dr.collect_parameters(self.du)
        self.dr.load_parameters(parameter_package)

    def test_set_up_hyperparameter_values(self):
        self.dr.set_up_hyperparameter_values()

    def test_create_shared_variables_h(self):
        initial_values = self.du.get('hemodynamic_parameter').astype(np.float32)
        self.dr.create_shared_variables_h(initial_values)

    '''
        def test_initializer_graph_forward_pass(self):
        du = self.du
        dr = self.dr

        # build model
        dr.build_an_initializer_graph()
        isess = tf.InteractiveSession()
        isess.run(tf.global_variables_initializer())

        # prepare data
        data = {'x': tb.split_data_for_initializer_graph(
                    du.get('x'), du.get('y'), dr.n_recurrent_step, dr.shift_x_y)[0],
                'y': tb.split_data_for_initializer_graph(
                    du.get('x'), du.get('y'), dr.n_recurrent_step, dr.shift_x_y)[1],
                'h': tb.split(du.get('h'), dr.n_recurrent_step, 0)}

        h_state_initial = dr.set_initial_hemodynamic_state_as_inactivated(dr.n_region).astype(np.float32)

        # run forward
        y_predicted, h_state_predicted = dr.run_initializer_graph(isess, h_state_initial, data['x'])

        # test
        np.testing.assert_array_almost_equal(
            np.array(np.concatenate(data['h'][:-1]), dtype=np.float32),
            np.array(h_state_predicted, dtype=np.float32))

        np.testing.assert_array_almost_equal(
            np.array(np.concatenate(data['y']), dtype=np.float32),
            np.array(y_predicted, dtype=np.float32),
            decimal=4)
    '''


    def test_mse(self):
        tf.reset_default_graph()
        t1 = tf.get_variable('tensor_1', shape=[100, 100])
        t2 = tf.get_variable('tensor_2', shape=[100, 100])
        tfm_mse_op = self.dr.mse(t1, t2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(16):
                v1 = np.random.rand(100, 100) * 10
                v2 = np.random.rand(100, 100) * 10
                tb_mse = tb.mse(v1, v2)
                tfm_mse = sess.run(tfm_mse_op, feed_dict={t1: v1, t2: v2})
                np.testing.assert_approx_equal(tb_mse, tfm_mse, significant=6)








