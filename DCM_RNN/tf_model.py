# This module contains the tensorflow model for DCM-RNN.
import tensorflow as tf
import numpy as np

class DcmRnn:
    def __init__(self, n_recurrent_step=None,
                 variable_scope_name_x=None, variable_scope_name_h=None):
        self.n_recurrent_step = n_recurrent_step or 12
        self.variable_scope_name_x = variable_scope_name_x or 'rnn_cell_x'
        self.variable_scope_name_h = variable_scope_name_h or 'rnn_cell_h'
        self.set_up_hyperparameter_values()

        self.trainable_flags_h = {'alpha': True,
                                  'E0': True,
                                  'k': True,
                                  'gamma': True,
                                  'tao': True,
                                  'epsilon': False,
                                  'V0': False,
                                  'TE': False,
                                  'r0': False,
                                  'theta0': False,
                                  'x_h_coupling': False
                                  }

    def collect_parameters(self, du):
        """
        Collect needed parameters from a DataUnit instant.
        :param du: a DataUnit instant
        :return: a dictionary containing needed paramters
        """
        deliverables = {}
        needed_parameters = {'n_node', 'n_stimuli', 't_delta'}
        for para in needed_parameters:
            deliverables[para] = du.get(para)
        return deliverables

    def load_parameters(self, parameter_package):
        self.n_region = parameter_package['n_node']
        self.t_delta = parameter_package['t_delta']
        self.n_stimuli = parameter_package['n_stimuli']

    def set_up_hyperparameter_values(self):
        hyperparameter_values = {}
        hyperparameter_values[self.variable_scope_name_x] = {'gradient': 1., 'sparse': 1., 'prior': 0.}
        hyperparameter_values[self.variable_scope_name_h] = {'gradient': 1., 'sparse': 0., 'prior': 1.}
        self.hyperparameter_values = hyperparameter_values


    def create_shared_variables_h(self, initial_values):
        """
        Create shared hemodynamic variables.
        :param initial_values: a pandas.DataFrame, containing the initial values of hemodynamic variables
        :return:
        """
        n_region, n_para = initial_values.shape
        with tf.variable_scope(self.variable_scope_name_h):
            for idx_r, region_label in enumerate(list(initial_values.index)):
                for para in initial_values.columns:
                    tf.get_variable(para + '_r' + str(idx_r),
                                      initializer=initial_values[para][region_label],
                                      trainable=self.trainable_flags_h[para])



    def build_an_initializer_graph(self, graph, parameter_package):
        """
        Build a model to estimate neural states from functional signal
        :param graph: a graph handle to draw tensorflow on
        :param parameter_package: a dictionary containing parameters needed
        :return: graph
        """
        # create variables


        pass