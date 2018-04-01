from unittest import TestCase
from dcm_rnn import toolboxes as tb
from dcm_rnn.tf_model import DcmRnn
import tensorflow as tf
import numpy as np
import os


class TestDcmRnn(TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.dr = DcmRnn()
        print(os.getcwd())
        data_path = 'dcm_rnn/resources/template0.pkl'
        self.du = tb.load_template(data_path)
        self.dr.collect_parameters(self.du)

    def test_collect_parameters(self):
        parameter_package = self.dr.collect_parameters(self.du)

    def test_load_parameters(self):
        parameter_package = self.dr.collect_parameters(self.du)
        self.dr.load_parameters(parameter_package)

    def test_set_up_hyperparameter_values(self):
        self.dr.set_up_loss_weights()

    def test_create_shared_variables_h(self):
        initial_values = self.du.get('hemodynamic_parameter').astype(np.float32)
        self.dr.create_shared_variables_h(initial_values)

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








