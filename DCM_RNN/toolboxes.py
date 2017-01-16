import random
import numpy as np
import pandas as pd
import math as mth
import scipy as sp
import scipy.stats
import os

class Initialization:
    def __init__(self, sparse_level=None, deviation_constraint=None, h_parameter_check_statistics=None,
                 x_init_low=None, x_init_high=None,
                 s_init_low=None, s_init_high=None,
                 f_init_low=None, f_init_high=None,
                 v_init_low=None, v_init_high=None,
                 q_init_low=None, q_init_high=None,
                 A_off_diagonal_low=None, A_off_diagonal_high=None,
                 A_diagonal_low=None, A_diagonal_high=None,
                 A_generation_max_trial_number=None,
                 B_init_low=None, B_init_high=None,
                 B_non_zero_probability=None,
                 B_sign_probability=None,
                 C_init_low=None, C_init_high=None
                 ):
        self.sparse_level = sparse_level or 0.5
        self.deviation_constraint = deviation_constraint or 1
        self.h_parameter_check_statistics = h_parameter_check_statistics or 'deviation'
        self.x_init_low = x_init_low or 0
        self.x_init_high = x_init_high or 0.4
        self.s_init_low = s_init_low or -0.3
        self.s_init_high = s_init_high or 0.3
        self.f_init_low = f_init_low or 0.8
        self.f_init_high = f_init_high or 1.4
        self.v_init_low = v_init_low or 0.8
        self.v_init_high = v_init_high or 1.4
        self.q_init_low = q_init_low or 0.6
        self.q_init_high = q_init_high or 1.

        self.A_off_diagonal_low = A_off_diagonal_low or -0.45
        self.A_off_diagonal_high = A_off_diagonal_high or 0.45
        self.A_diagonal_low = A_diagonal_low or -0.5
        self.A_diagonal_high = A_diagonal_high or 0
        self.A_generation_max_trial_number = A_generation_max_trial_number or 5000
        self.B_init_low = B_init_low or 0.2
        self.B_init_high = B_init_high or 0.5
        self.B_non_zero_probability = B_non_zero_probability or 0.5
        self.B_sign_probability = B_sign_probability or 0.5
        self.C_init_low = C_init_low or 0.5
        self.C_init_high = C_init_high or 1

        self.hemo_parameter_keys = ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0']
        self.hemo_parameter_mean = pd.Series([0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 100., 0.03, 25, 40.3],
                                             self.hemo_parameter_keys)
        self.hemo_parameter_variance = pd.Series([0.0015, 0.0024, 0.015, 0.002, 0.0568, 0., 0., 0., 0., 0.],
                                                 self.hemo_parameter_keys)

    def randomly_initialize_connection_matrices(self, n_node, n_stimuli, sparse_level=None):
        """
        Generate a set of matrices for neural level equation x'=Ax+\sigma(xBu)+Cu.
        For assumptions about each matrix, please refer to each individual generating function.
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :param sparse_level: sparse level of off_diagonal elements, [0, 1],
        actual non-zeros elements equal int(sparse_level * (n_node-1) * n_node)
        :return: a dictionary with elements 'A', 'B', and 'C'
        """
        sparse_level = sparse_level or self.sparse_level
        connection_matrices = {'A': self.randomly_generate_sparse_A_matrix(n_node, sparse_level),
                               'B': self.randomly_generate_B_matrix(n_node, n_stimuli),
                               'C': self.randomly_generate_C_matrix(n_stimuli, n_stimuli)}
        return connection_matrices

    def roll(self, probability):
        """
        sample a random number from [0,1), if it's smaller than or equal to given probability, return True,
        otherwise, False
        :param probability: a number from [0 ,1]
        :return: boolean
        """
        if random.random() >= probability:
            return True
        else:
            return False

    def check_transition_matrix(self, A):
        """
        Check whether all the eigenvalues of A have negative real parts
        to ensure a corresponding linear system is stable
        :param A: candidate transition matrix
        :return: True or False
        """
        w, v = np.linalg.eig(A)
        if max(w.real) < 0:
            return True
        else:
            return False

    def randomly_generate_A_matrix(self, n_node, sparse_level=None):
        """
        Generate the A matrix for neural level equation x'=Ax+\sigma(xBu)+Cu.
        Eigenvalues of A must to negative to ensure the system is stable.
        A is sparse on off-diagonal places
        Assumption:
        Diagonal elements are negative meaning self-feedback is negative.
        Strength range of diagonal elements is (-0.5, 0].
        Strength range of off-diagonal elements is [-0.45, 0.45)
        :param n_node: number of nodes (brain areas)
        :param sparse_level: sparse level of off_diagonal elements, [0, 1],
        actual non-zeros elements equal int(sparse_level * (n_node-1) * n_node)
        :return: a sparse A matrix
        """
        sparse_level = sparse_level or self.sparse_level

        def get_a_sparse_matrix(n_node, sparse_level):
            A = np.random.uniform(self.A_off_diagonal_low, self.A_off_diagonal_high, size=(n_node, n_node))
            A = (A * (1 - np.identity(n_node))).flatten()
            sorted_indexes = np.argsort(np.abs(A), None)  # ascending order
            kept_number = int(sparse_level * (n_node - 1) * n_node)
            A[sorted_indexes[list(range(0, len(A) - kept_number))]] = 0
            A = A.reshape([n_node, n_node])
            A = A * (1 - np.identity(n_node)) + \
                np.diag(np.random.uniform(self.A_diagonal_low, self.A_diagonal_high, size=n_node))
            return A

        if sparse_level > 1 or sparse_level < 0:
            raise ValueError('Imporper sparse_level, it should be [0,1]')
        A = get_a_sparse_matrix(n_node, sparse_level)
        count = 0
        while not self.check_transition_matrix(A):
            A = get_a_sparse_matrix(n_node, sparse_level)
            count += 1
            if count > self.A_generation_max_trial_number:
                raise ValueError('Can not generate qualified A matrix with max trail number!')
        return A

    def randomly_generate_B_matrix(self, n_node, n_stimuli):
        """
        Generate the B matrices for neural level equation x'=Ax+\sigma(xBu)+Cu.
        The number of B matrices equals the number of stimuli
        Assumptions:
        Each B matrix has only up to one non-zeros element, which means each stimulus only modify at most one edge
        Strength is in (-0.5, -0.2] U [0.2, 0.5)
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :return: a list of B matrices
        """
        if n_node < n_stimuli:
            raise ValueError('Improper number of nodes and stimuli')
        B = []
        indexes = random.sample(range(0, n_node ** 2), n_node ** 2)
        for matrix_index in range(n_stimuli):
            B_current = np.zeros((n_node, n_node))
            if self.roll(self.B_non_zero_probability):
                index_temp = indexes.pop()
                row_index = int(index_temp / n_node)
                column_index = index_temp % n_node
                sign = -1 if self.roll(self.B_sign_probability) else 1
                B_current[row_index, column_index] = sign * random.uniform(self.B_init_low, self.B_init_high)
            B.append(B_current)
        return B

    def randomly_generate_C_matrix(self, n_node, n_stimuli):
        """
        Generate the C matrix for neural level equation x'=Ax+\sigma(xBu)+Cu.
        Assumptions:
        Each stimulus functions on only one brain area.
        Strength is in [0.5, 1)
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :return: a C matrix
        """
        if n_node < n_stimuli:
            raise ValueError('Improper number of nodes and stimuli')
        node_indexes = random.sample(range(0, n_node), n_stimuli)
        C = np.zeros((n_node, n_stimuli))
        for culumn_index, row_index in enumerate(node_indexes):
            C[row_index, culumn_index] = np.random.uniform(self.C_init_low, self.C_init_high)
        return C

    def get_hemodynamic_parameter_prior_distributions(self):
        """
        Get prior distribution for hemodynamic parameters (Gaussian)
        :return: a pandas.dataframe containing mean, variance, and standard deviation of each parameter
        """
        prior_distribution = pd.DataFrame()
        prior_distribution['mean'] = self.hemo_parameter_mean
        prior_distribution['variance'] = self.hemo_parameter_variance
        prior_distribution['std'] = np.sqrt(self.hemo_parameter_variance)
        return prior_distribution.transpose()

    def get_expanded_hemodynamic_parameter_prior_distributions(self, n_node):
        """
        Repeat hemodynamic parameter prior distributions for each node and structure the results into pandas.dataframe
        :param n_node: number of nodes (brain areas)
        :return: a dict of  pandas.dataframe, containing hemodynamic parameter distributions for all the nodes,
                 distribution parameters include mean and standard deviation.
        """
        distributions = self.get_hemodynamic_parameter_prior_distributions()
        hemodynamic_parameters_mean = pd.DataFrame()
        hemodynamic_parameters_std = pd.DataFrame()
        for idx in range(n_node):
            hemodynamic_parameters_mean['region_' + str(idx)] = distributions.loc['mean']
            hemodynamic_parameters_std['region_' + str(idx)] = distributions.loc['std']
        return {'mean': hemodynamic_parameters_mean.transpose(), 'std': hemodynamic_parameters_std.transpose()}

    def get_standard_hemodynamic_parameters(self, n_node):
        """
        Get standard hemodynamic parameters, namely, the means of prior distribution of hemodynamic parameters
        :param n_node: number of nodes (brain areas)
        :return: a pandas data frame, containing hemodynamic parameters for all the nodes
        """
        return self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)['mean']

    def randomly_generate_hemodynamic_parameters(self, n_node, deviation_constraint=None):
        """
        Get random hemodynamic parameters, sampled from prior distribution.
        The sample range is constrained to mean +/- deviation_constraint * standard_deviation
        :param n_node: number of nodes (brain areas)
        :param deviation_constraint: float, used to constrain the sample range
        :return: a pandas data frame, containing hemodynamic parameters for all the nodes,
                 and optionally, normalized standard deviation.
        """
        deviation_constraint = deviation_constraint or self.deviation_constraint

        def sample_hemodynamic_parameters(hemodynamic_parameters_mean, hemodynamic_parameters_std,
                                          deviation_constraint):
            # sample a subject from hemodynamic parameter distribution
            h_mean = hemodynamic_parameters_mean
            h_std = hemodynamic_parameters_std

            h_para = h_mean.copy()
            h_devi = h_std.copy()

            p_shape = h_mean.shape
            for r in range(p_shape[0]):
                for c in range(p_shape[1]):
                    if h_std.iloc[r, c] > 0:
                        h_para.iloc[r, c] = np.random.normal(loc=h_mean.iloc[r, c], scale=h_std.iloc[r, c])
                        normalized_deviation = (h_para.iloc[r, c] - h_mean.iloc[r, c]) / h_std.iloc[r, c]
                        while abs(normalized_deviation) > deviation_constraint:
                            h_para.iloc[r, c] = np.random.normal(loc=h_mean.iloc[r, c], scale=h_std.iloc[r, c])
                            normalized_deviation = (h_para.iloc[r, c] - h_mean.iloc[r, c]) / h_std.iloc[r, c]
                    else:
                        pass
            return h_para

        temp = self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)
        hemodynamic_parameters_mean = temp['mean']
        hemodynamic_parameters_variance = temp['std']
        h_para = sample_hemodynamic_parameters(hemodynamic_parameters_mean,
                                               hemodynamic_parameters_variance,
                                               deviation_constraint)
        return h_para

    def check_hemodynamic_parameters(self, hemodynamic_parameters, h_parameter_check_statistics=None):
        """
        For each hemodynamic parameter, check whether it is sampled properly.
        :param hemodynamic_parameters: a pandas.dataframe, containing sampled hemodynamic parameters
        :param h_parameter_check_statistics: specify which kind of statistic to check,
               it takes value in {'deviation', 'pdf'}
        :return: a pandas.dataframe, containing checking statistics
        """
        h_parameter_check_statistics = h_parameter_check_statistics or self.h_parameter_check_statistics
        n_node = hemodynamic_parameters.shape[0]
        temp = self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)
        h_mean = temp['mean']
        h_std = temp['std']

        h_devi = h_mean.copy()
        h_pdf = h_mean.copy()
        h_para = hemodynamic_parameters

        p_shape = h_mean.shape
        for r in range(p_shape[0]):
            for c in range(p_shape[1]):
                if h_std.iloc[r, c] > 0:
                    h_devi.iloc[r, c] = (h_para.iloc[r, c] - h_mean.iloc[r, c]) / h_std.iloc[r, c]
                    h_pdf.iloc[r, c] = sp.stats.norm(0, 1).pdf(h_devi.iloc[r, c])
                else:
                    if h_para.iloc[r, c] == h_mean.iloc[r, c]:
                        h_devi.iloc[r, c] = 0.
                        h_pdf.iloc[r, c] = 1.
                    else:
                        raise ValueError('Hemodynamic parameter sampling goes wrong!')

        if h_parameter_check_statistics == 'deviation':
            return h_devi
        elif h_parameter_check_statistics == 'pdf':
            return h_pdf
        else:
            raise ValueError('Improper h_parameter_check_statistics value!')

    def if_proper_hemodynamic_parameters(self, hemodynamic_parameters, deviation_constraint=None):
        """
        Check if given hemodynamic parameters are within deviation constraint.
        If yes, return True; otherwise, throw error
        :param hemodynamic_parameters: a pandas.dataframe to be checked
        :param deviation_constraint: target deviation constraint, in normalized deviation,
        say if deviation_constraint=1, parameter with +/- one normalized deviation is acceptable
        :return: True or Error
        """
        deviation_constraint = deviation_constraint or self.deviation_constraint
        h_stat = self.check_hemodynamic_parameters(hemodynamic_parameters, h_parameter_check_statistics='deviation')
        max_deviation = abs(h_stat).values.max()
        if max_deviation <= deviation_constraint:
            return True
        else:
            raise ValueError('The given hemodynamic parameters are not qualified by given deviation_constraint')

    def set_initial_neural_state_as_zeros(self, n_node):
        """
        Set initial neural state as zeros.
        :param n_node: number of nodes (brain areas)
        :return: zeros
        """
        return np.zeros(n_node)

    def sample_initial_neural_state(self, n_node):
        """
        Sample initial neural state. The distribution is set by experience.
        :param n_node: number of nodes (brain areas)
        :return: sampled initial neural state
        """
        return np.random.uniform(self.x_init_low, self.x_init_high, size=n_node)

    def set_initial_hemodynamic_state_as_inactivated(self, n_node):
        """
        Set initial hemodynamic state as inactivated, namely, s=0, f=1, v=1, q=1.
        :param n_node: number of nodes (brain areas)
        :return: array of initial hemodynamic state
        """
        h_state_initial = np.ones((n_node, 4))
        h_state_initial[:, 0] = 0
        return h_state_initial

    def sample_initial_hemodynamic_state(self, n_node):
        """
        Sample initial hemodynamic state. The distribution is set by experience.
        :param n_node: number of nodes (brain areas)
        :return: array of initial hemodynamic state
        """
        h_state_initial = np.ones((n_node, 4))
        h_state_initial[:, 0] = np.random.uniform(low=self.s_init_low, high=self.s_init_high, size=n_node)
        h_state_initial[:, 1] = np.random.uniform(low=self.f_init_low, high=self.f_init_high, size=n_node)
        h_state_initial[:, 2] = np.random.uniform(low=self.v_init_low, high=self.v_init_high, size=n_node)
        h_state_initial[:, 3] = np.random.uniform(low=self.q_init_low, high=self.q_init_high, size=n_node)
        return h_state_initial

class ParameterGraph():
    def __init__(self):
        self.para_forerunner = {
            # level zero
            'if_random_neural_parameter': [],
            'if_random_hemodynamic_parameter': [],
            'if_random_x_state_initial': [],
            'if_random_h_state_initial': [],
            'if_random_stimuli': [],
            'if_random_node_number': [],
            'if_random_stimuli_number': [],
            'if_random_delta_t': [],
            'if_random_scan_time': [],

            # level one
            'initializer': ['if_random_neural_parameter',
                            'if_random_hemodynamic_parameter',
                            'if_random_x_state_initial',
                            'if_random_h_state_initial',
                            'if_random_stimuli',
                            'if_random_node_number',
                            'if_random_stimuli_number',
                            'if_random_delta_t',
                            'if_random_scan_time'],

            # level two
            'n_node': ['if_random_node_number', 'initializer'],
            't_delta': ['if_random_delta_t', 'initializer'],
            't_scan': ['if_random_scan_time', 'initializer'],

            # level three
            'n_time_point': ['t_scan', 't_delta'],
            'n_stimuli': ['if_random_stimuli_number', 'n_node', 'initializer'],

            'u': ['if_random_stimuli',
                  'n_stimuli',
                  'n_time_point',
                  'initializer'],

            'A': ['t_delta',
                  'if_random_neural_parameter',
                  'n_node',
                  'initializer'],
            'B': ['if_random_neural_parameter',
                  'n_node',
                  'n_stimuli',
                  'initializer'],
            'C': ['if_random_neural_parameter',
                  'n_node',
                  'n_stimuli',
                  'initializer'],

            'alpha': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'E0':['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'k':['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'gamma':['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'tao':['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'epsilon':['n_node','if_random_hemodynamic_parameter', 'initializer'],
            'V0':['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'TE':['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'r0':['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            'theta0':['n_node' ,'if_random_hemodynamic_parameter', 'initializer'],
            # they are all put in
            'hemodynamic_parameter': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],

            'initial_x_state': ['n_node', 'if_random_x_state_initial', 'initializer'],
            'initial_h_state': ['n_node', 'if_random_h_state_initial', 'initializer'],

            # level four, these matrices should never be assigned a value directly,
            # Use up level variables to generate them
            'Wxx': ['if_random_neural_parameter',
                    'n_node',
                    'A'],
            'Wxxu': ['if_random_neural_parameter',
                     'n_node',
                     'n_stimuli',
                     'B'],  # 'B' matrices equivalence in DCM_RNN model
            'Wx': ['if_random_neural_parameter',
                    'n_node',
                    'n_stimuli',
                    'C'],  # 'C' matrix equivalence in DCM_RNN model

            'Whh': ['hemodynamic_parameter', 't_delta'],
            'Whx': ['hemodynamic_parameter', 't_delta'],
            'bh': ['hemodynamic_parameter', 't_delta'],
            'Wo': ['hemodynamic_parameter'],
            'bo': ['hemodynamic_parameter'],


            # not necessary before estimation
            'n_backpro': [],  # number of truncated back propagation steps
            'learning_rate': [],  # used by tensorflow optimization operation
        }
        self.substitute_dictionary = {
            'alpha': 'hemodynamic_parameter',
            'E0': 'hemodynamic_parameter',
            'k': 'hemodynamic_parameter',
            'gamma': 'hemodynamic_parameter',
            'tao': 'hemodynamic_parameter',
            'epsilon': 'hemodynamic_parameter',
            'V0': 'hemodynamic_parameter',
            'TE': 'hemodynamic_parameter',
            'r0': 'hemodynamic_parameter',
            'theta0': 'hemodynamic_parameter'
        }

    def generate_gv_file(self):
        with open('documents/parameter_graph.gv', 'w') as f:
            f.write("digraph G {\n")
            f.write("          splines=ortho;\n")
            f.write("          rankdir = \"LR\";\n")
            f.write("          node[fontsize=24];\n")
            for key, value in self.para_forerunner.items():
                if not value:
                    # value is empty
                    string = "          " + key + ";\n"
                else:
                    # value is not empty
                    string = ""
                    for val in value:
                        if key not in self.substitute_dictionary.keys():
                            string = string + "          " + val + " -> " + key + ";\n"
                        else:
                            string = string + "          " + val + " -> " + self.substitute_dictionary[key] + ";\n"
                f.write(string)
            f.write("}")
        os.system('dot -Tpng documents/parameter_graph.gv -o documents/parameter_graph.png')