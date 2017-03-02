from unittest import TestCase
import tensorflow as tf
from DCM_RNN import toolboxes as tb
from DCM_RNN.tf_model import DcmRnn


class TestDcmRnn(TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.dr = DcmRnn()
        data_path = '../resources/template0.pkl'
        self.du = tb.load_template(data_path)


    def test_collect_parameters(self):
        parameter_package = self.dr.collect_parameters(self.du)

    def test_load_parameters(self):
        parameter_package = self.dr.collect_parameters(self.du)
        self.dr.load_parameters(parameter_package)

    def test_set_up_hyperparameter_values(self):
        self.dr.set_up_hyperparameter_values()

    def test_create_shared_variables_h(self):
        initial_values = self.du.get('hemodynamic_parameter')
        print(initial_values)
        self.dr.create_shared_variables_h(initial_values)

    def test_build_an_initializer_graph(self):
        # self.fail()
        pass


