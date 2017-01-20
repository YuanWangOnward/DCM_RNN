import random
import numpy as np
import pandas as pd
import math as mth
import scipy as sp
import scipy.stats
import os
import warnings
import collections


class OrderedDict(collections.OrderedDict):
    def __init__(self, dictionary=None, key_order=None):
        if dictionary == None and key_order == None:
            super().__init__()
        else:
            super().__init__()
            if isinstance(key_order, dict):
                if set(dictionary.keys()) != set(key_order.keys()):
                    raise ValueError('Dictionary and order do not match')
                else:
                    self.key_order = self.form_list_order(key_order)
                    sorted_key_order = sorted(key_order.items(), key=lambda item: item[1])
                    sorted_key_order = [item[0] for item in sorted_key_order]
                    for key in sorted_key_order:
                        self[key] = dictionary[key]
            elif isinstance(key_order, list):
                if set(dictionary.keys()) != set(key_order):
                    raise ValueError('Dictionary and order do not match')
                else:
                    self.key_order = key_order
                    for key in key_order:
                        self[key] = dictionary[key]
            else:
                raise ValueError('Key order should be a list or dictionary')

    def form_list_order(self, dict_order):
        """
        OrderDict accepts dictionary type order. Internally, it uses list order.
        :param dict_order: {key:order} dictionary
        :return: [keys] list
        """
        sorted_key_order = sorted(dict_order.items(), key=lambda item: item[1])
        sorted_key_order = [item[0] for item in sorted_key_order]
        return sorted_key_order

    def get_order_index(self, key):
        """
        Find index in the order for given key.
        :param key: key
        :return: index
        """
        if key not in self.key_order:
            raise ValueError(key + ' is not a valid element in key order')
        else:
            return self.key_order.index(key)


class Initialization:
    def __init__(self,
                 n_node_low=None, n_node_high=None,
                 stimuli_node_ratio=None,
                 t_delta_low=None, t_delta_high=None,
                 scan_time_low=None, scan_time_high=None,
                 x_init_low=None, x_init_high=None,
                 s_init_low=None, s_init_high=None,
                 f_init_low=None, f_init_high=None,
                 v_init_low=None, v_init_high=None,
                 q_init_low=None, q_init_high=None,
                 A_off_diagonal_low=None, A_off_diagonal_high=None,
                 A_diagonal_low=None, A_diagonal_high=None,
                 A_generation_max_trial_number=None,
                 sparse_level=None,
                 B_init_low=None, B_init_high=None,
                 B_non_zero_probability=None,
                 B_sign_probability=None,
                 C_init_low=None, C_init_high=None,
                 u_t_low=None, u_t_high=None,
                 deviation_constraint=None,
                 h_parameter_check_statistics=None
                 ):

        self.n_node_low = n_node_low or 3
        self.n_node_high = n_node_high or 11
        self.stimuli_node_ratio = stimuli_node_ratio or 1 / 3
        self.t_delta_low = t_delta_low or 0.05
        self.t_delta_high = t_delta_high or 0.5
        self.scan_time_low = scan_time_low or 3 * 60  # in second
        self.scan_time_high = scan_time_high or 10 * 60  # in second

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
        self.sparse_level = sparse_level or 0.5
        self.B_init_low = B_init_low or 0.2
        self.B_init_high = B_init_high or 0.5
        self.B_non_zero_probability = B_non_zero_probability or 0.5
        self.B_sign_probability = B_sign_probability or 0.5
        self.C_init_low = C_init_low or 0.5
        self.C_init_high = C_init_high or 1

        self.u_t_low = u_t_low or 1  # in second
        self.u_t_high = u_t_high or 10  # in second

        self.h_parameter_check_statistics = h_parameter_check_statistics or 'deviation'
        self.deviation_constraint = deviation_constraint or 1
        self.hemo_parameter_keys = ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0']
        self.hemo_parameter_mean = pd.Series([0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 100., 0.03, 25, 40.3],
                                             self.hemo_parameter_keys)
        self.hemo_parameter_variance = pd.Series([0.0015, 0.0024, 0.015, 0.002, 0.0568, 0., 0., 0., 0., 0.],
                                                 self.hemo_parameter_keys)

    def sample_node_number(self):
        """
        :return: n_node: number of nodes (brain areas)
        """
        return np.random.randint(self.n_node_low, self.n_node_high)

    def sample_stimuli_number(self, n_node, stimuli_node_ratio=None):
        """
        Return number of stimuli, n_stimuli is sampled from np.random.randint(1, 1 + int(n_node * stimuli_node_ratio))
        :param n_node: number of nodes (brain areas)
        :param stimuli_node_ratio: the max n_stimuli n_node ratio
        :return: number of stimuli
        """
        stimuli_node_ratio = stimuli_node_ratio or self.stimuli_node_ratio
        return np.random.randint(1, 1 + int(n_node * stimuli_node_ratio))

    def sample_t_delta(self):
        """
        :return: t_delta: time interval for approximate differential equations
        """
        return np.random.uniform(self.t_delta_low, self.t_delta_high)

    def sample_scan_time(self):
        """
        :return: total scan time in second
        """
        return np.random.uniform(self.scan_time_low, self.scan_time_high)

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

    def randomly_generate_u(self, n_stimuli, n_time_point, t_delta):
        """
        Randomly generate input stimuli,
        :param n_stimuli: the amount of stimuli
        :param n_time_point: the total sample point of stimuli
        :param t_delta: time interval between adjacent sample points
        :return: np.array, random input, size (n_stimuli, n_time_point)
        """

        def flip(num):
            if num is 0:
                return 1
            if num is 1:
                return 0

        u_t_low = self.u_t_low
        u_t_high = self.u_t_high
        u_n_low = int(u_t_low / t_delta)
        u_n_high = int(u_t_high / t_delta)
        u = np.zeros((n_stimuli, n_time_point))

        for n_s in range(n_stimuli):
            i_current = 0
            value = 0
            while i_current < n_time_point:
                step = np.random.randint(u_n_low, u_n_high)
                value = flip(value)
                i_next = i_current + step
                if i_next >= n_time_point:
                    i_next = n_time_point
                u[n_s, i_current:i_next] = value
                i_current = i_next
        return u

    def get_impulse_u(self, n_stimuli, n_time_point):
        """
        Generate impulse stimuli.
        :param n_stimuli: the amount of stimuli
        :param n_time_point: the total sample point of stimuli
        :return: np.array, size (n_stimuli, n_time_point)
        """
        u = np.zeros((n_stimuli, n_time_point))
        u[:, 0] = 1
        return u

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

    def evaluate_hemodynamic_parameters(self, hemodynamic_parameters, h_parameter_check_statistics=None):
        """
        For each hemodynamic parameter, evaluate how it derives from the means.
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

    def check_proper_hemodynamic_parameters(self, hemodynamic_parameters, deviation_constraint=None):
        """
        Check if given hemodynamic parameters are within deviation constraint.
        If yes, return True; otherwise, throw error
        :param hemodynamic_parameters: a pandas.dataframe to be checked
        :param deviation_constraint: target deviation constraint, in normalized deviation,
        say if deviation_constraint=1, parameter with +/- one normalized deviation is acceptable
        :return: True or Error
        """
        deviation_constraint = deviation_constraint or self.deviation_constraint
        h_stat = self.evaluate_hemodynamic_parameters(hemodynamic_parameters, h_parameter_check_statistics='deviation')
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

    def randomly_generate_initial_neural_state(self, n_node):
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

    def randomly_generate_initial_hemodynamic_state(self, n_node):
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

    def calculate_dcm_rnn_x_matrices(self, A, B, C, t_delta):
        """
        Calculate matrices used in DCM_RNN for neural activity evolving.
        In DCM, neural level equation x'=Ax+\sigma(xBu)+Cu.
        In DCM_RNN, neural level equation x_t+1 = Wxx * x_t + \sigma(x_t * Wxxu * u) + Wxu * u
        :param A:
        :param B:
        :param C:
        :param t_delta: time interval for approximate differential equations in second
        :return: {'Wxx': Wxx, 'Wxxu': Wxxu, 'Wxu': Wxu}
        """
        n_node = A.shape[0]
        Wxx = np.array(A) * t_delta + np.eye(n_node, n_node, 0)
        Wxxu = [np.array(b) * t_delta for b in B]
        Wxu = C * t_delta
        return {'Wxx': Wxx, 'Wxxu': Wxxu, 'Wxu': Wxu}

    def calculate_dcm_rnn_h_matrices(self, hemodynamic_parameter, t_delta):
        """
        Calculate matrices used in DCM_RNN for hemodynamic evolving.
        :param hemodynamic_parameter: pd.dataframe
        :param t_delta: time interval for approximate differential equations in second
        :return: {'Whh': [Whh], 'Whx': [Whx], 'bh': [bh], 'Wo': [Wo], 'bo': [bo]}
        """
        n_node = hemodynamic_parameter.shape[0]
        Whh_ = []
        Whx_ = []
        Wo_ = []
        bh_ = []
        bo_ = []

        for n in range(n_node):
            # ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0']
            E0 = hemodynamic_parameter.loc['region_' + str(n), 'E0']
            k = hemodynamic_parameter.loc['region_' + str(n), 'k']
            gamma = hemodynamic_parameter.loc['region_' + str(n), 'gamma']
            tao = hemodynamic_parameter.loc['region_' + str(n), 'tao']
            epsilon = hemodynamic_parameter.loc['region_' + str(n), 'epsilon']
            V0 = hemodynamic_parameter.loc['region_' + str(n), 'V0']
            TE = hemodynamic_parameter.loc['region_' + str(n), 'TE']
            r0 = hemodynamic_parameter.loc['region_' + str(n), 'r0']
            theta0 = hemodynamic_parameter.loc['region_' + str(n), 'theta0']

            Whh = np.zeros((4, 7))
            Whh[0, 0] = -(t_delta * k - 1)
            Whh[0, 1] = -t_delta * gamma
            Whh[1, 0] = t_delta
            Whh[1, 1] = 1
            Whh[2, 1] = t_delta / tao
            Whh[2, 2] = 1
            Whh[2, 4] = -t_delta / tao
            Whh[3, 3] = 1
            Whh[3, 5] = -t_delta / tao
            Whh[3, 6] = t_delta / tao
            Whh_.append(Whh)

            Whx_.append(np.array([t_delta, 0, 0, 0]).reshape(4, 1))

            Wo = np.array([-(1 - epsilon) * V0, -4.3 * theta0 * E0 * V0 * TE, -epsilon * r0 * E0 * V0 * TE])
            Wo_.append(Wo)

            bh = np.array(np.asarray([t_delta * gamma, 0, 0, 0]).reshape(4, 1))
            bh_.append(bh)

            bo = V0 * (4.3 * theta0 * E0 * TE + epsilon * r0 * E0 * TE + (1 - epsilon))
            bo_.append(bo)


        return {'Whh': Whh_, 'Whx': Whx_, 'bh': bh_, 'Wo': Wo_, 'bo': bo_}


class ParameterGraph:
    def __init__(self):
        self._para_forerunner = {
            # inherited
            'initializer': [],

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

            #
            'n_node': ['if_random_node_number', 'initializer'],
            't_delta': ['if_random_delta_t', 'initializer'],
            't_scan': ['if_random_scan_time', 'initializer'],

            #
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

            # 'alpha': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'E0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'k': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'gamma': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'tao': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'epsilon': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'V0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'TE': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'r0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'theta0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # they are all put in

            'hemodynamic_parameter': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],

            'initial_x_state': ['n_node', 'if_random_x_state_initial', 'initializer'],
            'initial_h_state': ['n_node', 'if_random_h_state_initial', 'initializer'],

            'Wxx': ['A', 't_delta'],
            'Wxxu': ['B', 't_delta'],
            'Wxu': ['C', 't_delta'],  # 'C' matrix equivalence in DCM_RNN model

            'Whh': ['hemodynamic_parameter', 't_delta'],
            'Whx': ['hemodynamic_parameter', 't_delta'],
            'bh': ['hemodynamic_parameter', 't_delta'],
            'Wo': ['hemodynamic_parameter'],
            'bo': ['hemodynamic_parameter'],

            # not necessary before estimation
            'n_backpro': [],  # number of truncated back propagation steps
            'learning_rate': [],  # used by tensorflow optimization operation
        }

        self.para_substitute = {
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
        }  # not in use now

        level_para = {
            'inherited': ['initializer'],
            'level_0': ['if_random_neural_parameter',
                        'if_random_hemodynamic_parameter',
                        'if_random_x_state_initial',
                        'if_random_h_state_initial',
                        'if_random_stimuli',
                        'if_random_node_number',
                        'if_random_stimuli_number',
                        'if_random_delta_t',
                        'if_random_scan_time',
                        'n_backpro',
                        'learning_rate'],
            'level_1': ['n_node', 't_delta', 't_scan'],
            'level_2': ['n_time_point',
                        'n_stimuli',
                        'A',
                        'hemodynamic_parameter',
                        'initial_x_state',
                        'initial_h_state'],
            'level_3': ['u',
                        'B', 'C',
                        'Wxx', 'Whx', 'Whh', 'bh', 'Wo', 'bo'],
            'level_4': ['Wxxu', 'Wxu']
        }
        level_para_order = ['inherited', 'level_0', 'level_1', 'level_2', 'level_3', 'level_4']
        self._level_para = OrderedDict(level_para, level_para_order)

        self.check_parameter_relation()

    def check_parameter_relation(self):
        """
        Check if _para_forerunner and para_level are in consistence with each other, since they are hand-coded
        which is prone to error
        :return: True or raiseError
        """

        variable_level_map = {}
        for key, value in self._level_para.items():
            for val in value:
                variable_level_map[val] = self._level_para.get_order_index(key)
        for key, value in self._para_forerunner.items():
            if not value:
                if variable_level_map[key] not in [0, 1]:
                    raise ValueError(key + ' parameter graph error')
            else:
                temp = [variable_level_map[val] for val in value]
                max_forerunner_level = max(temp)
                if key in self.para_substitute.keys():
                    key = self.para_substitute[key]
                if variable_level_map[key] <= max_forerunner_level:
                    raise ValueError(key + ' parameter graph error')
        return True

    def get_para_forerunner_mapping(self):
        return self._para_forerunner.copy()

    def forerunner2descendant(self, forerunner_mapping=None, key_order=None, if_complete=True):
        """
        Transfer {parameter:[forerunners]} mapping  dictionary to
        {parameter:[descendants]} mapping dictionary
        :param forerunner_mapping: parameter:forerunners mapping dictionary
        :param key_order: if an {key:order} dictionary is given, return an OderderDict, otherwise, a dict
        :param if_complete: if True, treat keys in forerunner dictionary as valid parameters.
            Parameter without a descendant will also be added to resulting dictionary
        :return: parameter: {parameter:[descendants]} mapping dictionary
        """
        forerunner_mapping = forerunner_mapping or self._para_forerunner
        descendant_mapping = {}
        for key, value in forerunner_mapping.items():
            for val in value:
                if val in descendant_mapping.keys():
                    descendant_mapping[val].append(key)
                else:
                    descendant_mapping[val] = [key]
        if if_complete is True:
            all_name = []
            for key, value in forerunner_mapping.items():
                all_name = all_name + [key]
                all_name = all_name + value
            all_name = set(all_name)
            for name in all_name:
                if name not in descendant_mapping.keys():
                    descendant_mapping[name] = []
        if key_order != None:
            if set(key_order.keys()) != set(descendant_mapping.keys()):
                raise ValueError('Given order does not match reversed dictionary.')
            else:
                return OrderedDict(descendant_mapping, key_order)
        else:
            return descendant_mapping

    def get_para_descendant_mapping(self):
        para_forerunner = self.get_para_forerunner_mapping()
        para_descendant = self.forerunner2descendant(para_forerunner)
        return para_descendant

    def level_para2para_level(self, level_para=None):
        """
        Transfer {level:[parameters]} mapping dictionary to
        {parameter:level} mapping dictionary
        :param level_para: {level:[parameters]} mapping dictionary
        :return: {parameter:level} mapping dictionary, if _level_para is an OrderedDict, return an OrderedDict
        """
        level_para = level_para or self._level_para
        para_level_temp = self.forerunner2descendant(level_para, if_complete=False)
        if isinstance(level_para, OrderedDict):
            para_level = OrderedDict()
            key_order = []
            for level in level_para.key_order:
                for para, value in para_level_temp.items():
                    if value[0] == level:
                        para_level[para] = level
                        key_order.append(para)
            para_level.key_order = key_order
        else:
            para_level = {}
            for key, value in para_level_temp.items():
                if len(value) > 1:
                    raise ValueError('One parameters cannot belong to more than one level.')
                else:
                    para_level[key] = value[0]
        return para_level

    def get_level_para_mapping(self):
        return self._level_para.copy()

    def get_para_level_mapping(self, level_para=None):
        """
        Get {parameter:level} dictionary from {level:[parameters]} dictionary
        :param level_para:{level:[parameters]} dictionary
        :return: {parameter:level} dictionary
        """
        level_para = level_para or self._level_para
        return self.level_para2para_level(level_para)

    def get_para_level_index_mapping(self, level_para=None, order=None):
        """
        Get {parameter:level_index} dictionary from {level:[parameters]} dictionary
        :param level_para:{level:[parameters]} dictionary
        :param order: [levels] list recording level order
        :return: {parameter:level} dictionary
        """
        level_para = level_para or self._level_para
        order = order or self._level_para.key_order
        para_level = self.get_para_level_mapping(level_para)
        return {key: order.index(value) for key, value in para_level.items()}

    def get_para_category_mapping(self, para_forerunner=None):
        """
        Base on prerequisites (forerunners), put parameters in to 3 categories.
        Category one: no prerequisites, should be assigned value directly
        Category two: with prerequisite and has if_random_ flag in prerequisites, one needs to check flag before assign
        Category three: with prerequisite and has no flag, should not be assigned directly, its value should be derived
            by its prerequisites.
        :param para_forerunner: a dictionary recording _para_forerunner of each parameter
        :return: a {para:category} dictionary recording category of each parameter
        """
        para_category = {}
        para_forerunner = para_forerunner or self._para_forerunner
        for key, value in para_forerunner.items():
            if not value:
                para_category[key] = 1
            else:
                flag = self.abstract_flag(value)
                if flag != None:
                    para_category[key] = 2
                else:
                    para_category[key] = 3
        return para_category

    def para_category2category_para(self, para_category, category_order=None):
        """
        Transform {para:category} mapping to {category:[paras]} mapping
        :param para_category: {para:category} mapping
        :param category_order: order of the category, if not specified, sorted() will be used
        :return: {category:[paras]} mapping, OrderedDict
        """
        category_para_temp = {}
        for key, value in para_category.items():
            if value in category_para_temp.keys():
                category_para_temp[value].append(key)
            else:
                category_para_temp[value] = [key]
        if category_order is None:
            category_order = sorted(category_para_temp.keys())
        category_para = OrderedDict(category_para_temp, category_order)
        return category_para

    def get_category_para_mapping(self, para_forerunner=None):
        para_forerunner = para_forerunner or self._para_forerunner
        para_category = self.get_para_category_mapping(para_forerunner)
        category_para = self.para_category2category_para(para_category)
        return category_para

    def abstract_flag(self, prerequisites):
        """
        Find if_ flag from the prerequisites of a parameter
        :param prerequisites: a list of parameters
        :return: None, if prerequisites is empty, or there is no flag in prerequisites
                 a flag name if there is a flag in prerequisites
        """
        if not prerequisites:
            return None
        else:
            temp = [s for s in prerequisites if 'if_' in s]
            if len(temp) == 0:
                return None
            elif len(temp) == 1:
                return temp[0]
            else:
                print(prerequisites)
                raise ValueError('Multiple flags found.')

    def get_flag_name(self, para):
        """
         Find if_ flag from the forerunners of a parameter
        :param para: target parameter
        :return: None, if forerunners is empty, or there is no flag in forerunners
                 a flag name if there is a flag in forerunners
        """
        self.check_valid_para(para)
        para_forerunner = self.get_para_forerunner_mapping()
        return self.abstract_flag(para_forerunner[para])

    def make_graph(self, relation_dict=None, file_name=None, rank_dict=None, rank_order=None):
        """
        Create .gv file and then use dot tool to create a diagram.
        :param relation_dict: {form:[tos]} structure, recording edges
        :param file_name: title, file name of .gv and .png file
        :param rank_dict: {rank_name:[members]} dict recording rank information
        :param rank_order: [rank_names] list recording rank order
        :return: True if it runs to the end
        """
        if relation_dict == None and file_name == None and rank_dict == None and rank_order == None:
            relation_dict = self.forerunner2descendant(self._para_forerunner)
            file_name = "parameter_level_graph"
            rank_dict = self._level_para
            rank_order = self._level_para.key_order

        with open("documents/" + file_name + ".gv", 'w') as f:
            f.write("digraph G {\n")
            f.write("          splines=ortho;\n")
            f.write("          fontsize = 48;\n")
            f.write("          rankdir = \"LR\";\n")
            f.write("          node[fontsize=24];\n")
            f.write("          edge[penwidth=2];\n")

            if rank_order != None:
                f.write("          {\n")
                f.write("          node [shape=plaintext fontsize=36];\n")
                for key in rank_order:
                    if key != rank_order[-1]:
                        f.write("          " + str(key) + " -> \n")
                    else:
                        f.write("          " + str(key) + "\n")
                f.write("          }\n")

            if rank_dict != None:
                for key, value in rank_dict.items():
                    f.write("          {rank = same;\n")
                    if type(key) is not str:
                        key = str(key)
                    f.write("          " + key + ";\n")
                    for val in value:
                        f.write("          " + val + ";\n")
                    f.write("          }\n")

            # relation edges
            for key, value in relation_dict.items():
                randome_coloar = ''.join([random.choice('0123456789ABCE') for x in range(6)])
                if not value:
                    # value is empty

                    string = "          " + key + " ;\n"
                else:
                    # value is not empty
                    string = ""
                    for val in value:
                        string = string + "          " + key + " -> " + val + " [color=\"#" + randome_coloar + "\"];\n"
                f.write(string)

            # add title
            f.write('          labelloc = "t";\n')
            f.write('          label = "' + file_name + '";\n')
            f.write("}")
            # call dot tool to draw diagram
            # however, it doesn't work, so removed for the moment
            # source_file = "documents/" + file_name + ".gv"
            # target_file = "documents/" + file_name + ".png"
            # subprocess.run(["dot", "-Tpng", source_file, "-o", target_file], check=True)

    def get_all_para_names(self):
        """
        Get all parameter names in a DataUnit.
        :return: a list of parameter names, sorted by parameter level
        """
        para_level = self.get_para_level_mapping()
        return sorted(para_level.keys(), key=lambda key: para_level[key])

    def check_valid_para(self, para):
        """
        Check if a given parameter name is a valid one
        :param para: target name
        :return: True or raise error
        """
        valid_names = self.get_all_para_names()
        if para in valid_names:
            return True
        else:
            raise ValueError('Improper name.')


class DataUnit(Initialization, ParameterGraph):
    """
    This class is used to ensure consistence and integrity of all data, but that takes a lot of efforts, so currently,
    it's used in a unsecured manner. Namely, DataUnit inherits dict and it key/value pair can be changed without other
    constraints, which means they might not be consistent.
    A internal dictionary, _secured_data is a dictionary should only manipulated by internal methods. It's not
    implemented currently but DataUnit should keep it structure so that this functionality can be added easily.
    """
    init = Initialization()

    def __init__(self,
                 if_random_neural_parameter=True,
                 if_random_hemodynamic_parameter=True,
                 if_random_x_state_initial=True,
                 if_random_h_state_initial=True,
                 if_random_stimuli=True,
                 if_random_node_number=False,
                 if_random_stimuli_number=True,
                 if_random_delta_t=False,
                 if_random_scan_time=False,
                 ):
        Initialization.__init__(self)
        ParameterGraph.__init__(self)
        self._secured_data = {}
        self._secured_data['if_random_neural_parameter'] = if_random_neural_parameter
        self._secured_data['if_random_hemodynamic_parameter'] = if_random_hemodynamic_parameter
        self._secured_data['if_random_x_state_initial'] = if_random_x_state_initial
        self._secured_data['if_random_h_state_initial'] = if_random_h_state_initial
        self._secured_data['if_random_stimuli'] = if_random_stimuli
        self._secured_data['if_random_node_number'] = if_random_node_number
        self._secured_data['if_random_stimuli_number'] = if_random_stimuli_number
        self._secured_data['if_random_delta_t'] = if_random_delta_t
        self._secured_data['if_random_scan_time'] = if_random_scan_time
        self._secured_data['initializer'] = self

        # when do auto data generating, following the order below
        self._assign_order = ['n_node',
                              'n_stimuli',
                              't_scan',
                              't_delta',
                              'n_time_point',
                              'A', 'B', 'C', 'u',
                              'hemodynamic_parameter',
                              'initial_x_state',
                              'initial_h_state',
                              'Wxx', 'Wxxu', 'Wxu',
                              'Whh', 'Whx', 'bh',
                              'Wo', 'bo']

    def get_assign_order(self):
        return self._assign_order.copy()

    def set(self, para, value):
        """
        Set value to a parameter.
        A parameter can only be directly assigned a number if it is of category one or two (see ParameterGraph)
        When a parameter is assigned a number, its compatibility should be checked with existing parameters.
        But it takes a lot of time to implement.
        Now it only check if_random flag and descendant.
        If if_random flag is True, or if its descendants have had value, it should not be assigned.
        :param para: name of target parameter
        :param value: value to be assigned
        :return: True if successful
        """
        self.check_valid_para(para)
        para_category = self.get_para_category_mapping()
        category = para_category[para]
        if category is 3:
            raise ValueError('Category 3 parameter should not be assigned directly')
        elif category is 1:
            self.check_has_no_assigned_descendant(para)
            self._secured_data[para] = value
        elif category is 2:
            self.check_has_no_assigned_descendant(para)
            flag_name = self.get_flag_name(para)
            if flag_name is None:
                self._secured_data[para] = value
            elif flag_name in self._secured_data.keys():
                if self._secured_data[flag_name] is False:
                    self._secured_data[para] = value
                else:
                    raise ValueError(flag_name + 'is True, ' + para + 'cannot be assigned directly')
            else:
                raise ValueError(flag_name + 'has not been assigned, ' + para + 'cannot be assigned directly')
        else:
            raise ValueError('Category error.')

    def if_has_no_assigned_descendant(self, para):
        """
        Check the descendants of a para, if any of them has a value, return False
        :param para: target parameter
        :return: If no descendant has had value, return True; otherwise, False.
        """
        self.check_valid_para(para)
        para_descendant = self.get_para_descendant_mapping()
        descendants = para_descendant[para]
        if not descendants:
            return True
        else:
            flags = [True if value in self._secured_data.keys() else False for value in descendants]
            if True in flags:
                return False
            else:
                return True

    def check_has_no_assigned_descendant(self, para):
        """
        Check the descendants of a para, if any of them has a value, raise error
        :param para: target parameter
        :return: If no descendant has had value, return True; otherwise, error.
        """
        flag = self.if_has_no_assigned_descendant(para)
        if flag:
            return True
        else:
            raise ValueError(para + 'has descendant with value')

    def if_has_value(self, para):
        """
        Check whether if a parameter has been assigned a value
        :param para: target para
        :return: True if para has a value; otherwise False
        """
        self.check_valid_para(para)
        if para in self._secured_data.keys():
            return True
        else:
            return False

    def complete_data_unit(self):
        """
        With given category_one parameters, generate all missing ones
        :return: True
        """
        # check category one parameters
        cate_para = self.get_category_para_mapping()
        para_cate_one = cate_para[1]
        flags = [True if para in self._secured_data.keys() else False for para in para_cate_one]
        if False in flags:
            parameters = [para_cate_one[index] for index, value in enumerate(flags) if value is False]
            string = ', '.join(parameters)
            raise ValueError(string + " has (have) not been specified.")
        # assign parameter one by one follow self._assign_order
        assign_order = self.get_assign_order()
        for para in assign_order:
            self.check_has_no_assigned_descendant(para)
            flag_name = self.get_flag_name(para)
            self._set(para, flag_name)

    def _set(self, para, flag_name):
        """
        Used by auto data generating process.
        Assume all constraints have been checked, and value assignment is allowed.
        Generate value for para following flag information.
        :param para: target para
        :param flag_name: if_random flag
        :return:
        """

        def show(para_name, flag_name, flag_value, method):
            if flag_name is None:
                flag_name = 'None'
            message = flag_name + ' is ' + str(flag_value) + ', ' + para_name + ' is set by ' + method
            print(message)
        def raise_error(para_name, flag_name, flag_value, error):
            message = flag_name + ' is ' + str(flag_value) + ', ' + para_name + ' ' + error
            raise ValueError(message)


        assign_order = ['n_node',
                        'n_stimuli',
                        't_scan',
                        't_delta',
                        'n_time_point',
                        'A', 'B', 'C', 'u',
                        'hemodynamic_parameter',
                        'initial_x_state',
                        'initial_h_state',
                        'Wxx', 'Wxxu', 'Wxu',
                        'Whh', 'Whx', 'bh',
                        'Wo', 'bo']
        if flag_name is not None:
            flag_value = self._secured_data[flag_name]
        else:
            flag_value = 'none_flag'
        if para is 'n_node':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_node_number')
                self._secured_data[para] = self.sample_node_number()
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'n_stimuli':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_stimuli_number')
                self._secured_data[para] = self.sample_stimuli_number(self._secured_data['n_node'])
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 't_scan':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_scan_time')
                self._secured_data[para] = self.sample_scan_time()
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 't_delta':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_t_delta')
                self._secured_data[para] = self.sample_t_delta()
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'n_time_point':
            assert flag_name is None
            show(para, flag_name, flag_value, 'calculation')
            para_temp = int(self._secured_data['t_scan'] / self._secured_data['t_delta'])
            para_temp = mth.ceil(para_temp / 32) * 32
            self._secured_data[para] = para_temp
        elif para is 'A':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_A_matrix')
                self._secured_data[para] = self.randomly_generate_A_matrix(self._secured_data['n_node'])
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'B':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_B_matrix')
                self._secured_data[para] = self.randomly_generate_B_matrix(self._secured_data['n_node'],
                                                                           self._secured_data['n_stimuli'])
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'C':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_C_matrix')
                self._secured_data[para] = self.randomly_generate_C_matrix(self._secured_data['n_node'],
                                                                           self._secured_data['n_stimuli'])
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'u':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_u')
                self._secured_data[para] = self.randomly_generate_u(self._secured_data['n_stimuli'],
                                                                    self._secured_data['n_time_point'],
                                                                    self._secured_data['t_delta'])
            else:
                show(para, flag_name, flag_value, 'get_impulse_u')
                self._secured_data[para] = self.get_impulse_u(self._secured_data['n_stimuli'],
                                                              self._secured_data['n_time_point'])
        elif para is 'hemodynamic_parameter':
            if flag_value is True:
                show(para, flag_name, flag_value, 'get_impulse_u')
                self._secured_data[para] = self.randomly_generate_hemodynamic_parameters(self._secured_data['n_node'])
            else:
                show(para, flag_name, flag_value, 'get_standard_hemodynamic_parameters')
                self._secured_data[para] = self.get_standard_hemodynamic_parameters(self._secured_data['n_node'])
        elif para is 'initial_x_state':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_initial_neural_state')
                self._secured_data[para] = self.randomly_generate_initial_neural_state(self._secured_data['n_node'])
            else:
                show(para, flag_name, flag_value, 'set_initial_neural_state_as_zeros')
                self._secured_data[para] = self.set_initial_neural_state_as_zeros(self._secured_data['n_node'])
        elif para is 'initial_h_state':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_initial_hemodynamic_state')
                self._secured_data[para] = \
                    self.randomly_generate_initial_hemodynamic_state(self._secured_data['n_node'])
            else:
                show(para, flag_name, flag_value, 'set_initial_hemodynamic_state_as_inactivated')
                self._secured_data[para] = \
                    self.set_initial_hemodynamic_state_as_inactivated(self._secured_data['n_node'])
        elif para in ['Wxx', 'Wxxu', 'Wxu']:
            assert flag_name is None
            show(para, flag_name, flag_value, 'calculation')
            para_temp = self.calculate_dcm_rnn_x_matrices(self._secured_data['A'],
                                                          self._secured_data['B'],
                                                          self._secured_data['C'],
                                                          self._secured_data['t_delta'])
            self._secured_data[para] = para_temp[para]







    def call_uniformed_assignment_api(self, parameter, value=None, tag='random'):
        """
        Using different method to assign value to parameter according to the parameter category.
        See details in get_para_category_mapping()
        :param parameter: the target variable to be assigned
        :param value: if a value is needed for the assignment, use it
        :param tag: control value generating process rather than providing a detailed value, like 'random' or 'standard'
        :return: null, it adds element into DataUnit._secured_data
        """
        if parameter not in self.para_categories.keys():
            raise ValueError(parameter + " is not a valid parameter name")
        else:
            category = self.para_categories[parameter]
            if category == 1:
                pass
                # no prerequisites, should be assigned value directly

        prerequisites = self.para_prerequisites[parameter]
        if not prerequisites:
            if value != None:
                self._secured_data[parameter] = value
            else:
                raise ValueError("Please specify a value for " + parameter)
        else:
            # have prerequisites
            flag = self.abstract_flag(prerequisites)
            if flag == None:
                # without flag
                if value != None:
                    warnings.warn(parameter + "should not be spesified manually, value is ignored.")
                prerequisites_check = [para in self._secured_data.keys() for para in prerequisites]
                if False in prerequisites_check:
                    not_satisfied_prerequistes = [prerequisites for val in prerequisites_check if not val]
                    raise ValueError(parameter + " cannot be assigned because the following prerequisites have not be "
                                     + "assigned: " + not_satisfied_prerequistes)
                else:
                    # if all prerequisites have been assigned, generate values
                    #
                    pass
            else:
                # with flag
                if self._secured_data[flag] == True:
                    # generate randomly
                    # check prerequisites
                    if 'n_node' in self._secured_data.keys():
                        n_stimuli = super().sample_stimuli_number(self._secured_data['n_node'])
                        self._secured_data['n_stimuli'] = n_stimuli
                    else:
                        raise ValueError('n_node has not be specified.')
                else:
                    # assign a value
                    if value == None:
                        raise ValueError(parameter + " needs a value.")
                    else:
                        self._secured_data[parameter] = value

    def check_prerequisites(self, parameter):
        """
        Check if all prerequisites of a parameter have been specified
        :param parameter: target parameter
        :return: True or raise error
        """
        prerequisites_check = [para in self._secured_data.keys() for para in self.para_prerequisites[parameter]]
        if False in prerequisites_check:
            not_satisfied_prerequisites = [self.para_prerequisites[parameter] for val in prerequisites_check if not val]
            string = ''
            for value in not_satisfied_prerequisites:
                string = value + ' ' + string
            raise ValueError(parameter + " cannot be assigned because the following prerequisites have not be "
                             + "assigned: " + string)
        else:
            return True

    def set_category_one_parameter(self, parameter, value):
        """
        Set value of category one parameter. Directly set.
        :param parameter: name of parameter
        :param value: value of parameter
        :return: None
        """
        self._secured_data[parameter] = value

    def set_category_two_parameter(self, parameter, value, tag):
        """
        Set value of category two parameter.
        All its prerequisites should be checked.
        For simplicity, only existence of prerequisites are checked here.
        :param parameter: name of parameter
        :param value: value of parameter
        :param tag: control value generating process rather than providing a detailed value, like 'random' or 'standard'
        :return:
        """
        # check if all prerequisites exis
        if self.check_prerequisites(parameter):
            flag = self.abstract_flag(self.para_prerequisites[parameter])
            if self._secured_data[flag] == False:
                if value != None:
                    self._secured_data[parameter] = value
                else:
                    raise ValueError("A proper value is needed for  " + parameter)
            else:
                if value != None:
                    warnings.warn(parameter + "should not be specified manually since " + flag + " is True. "
                                                                                                 "Value is ignored.")
                # here list all category two parameters and call according functions
                if parameter == 'A':
                    self._secured_data[parameter] = self.randomly_generate_A_matrix(self._secured_data['n_node'])
                elif parameter == 'B':
                    self._secured_data[parameter] = self.randomly_generate_B_matrix(self._secured_data['n_node'],
                                                                                    self._secured_data['n_stimuli'])
                elif parameter == 'C':
                    self._secured_data[parameter] = self.randomly_generate_C_matrix(self._secured_data['n_node'],
                                                                                    self._secured_data['n_stimuli'])
                elif parameter == 'initial_x_state':
                    if tag == 'random':
                        self._secured_data[parameter] = self.randomly_generate_initial_neural_state(self._secured_data['n_node'])
                    elif tag == 'standard':
                        self._secured_data[parameter] = \
                            self.set_initial_neural_state_as_zeros(self._secured_data['n_node'])
                    else:
                        raise ValueError('Improper tag')
                elif parameter == 'initial_h_state':
                    if tag == 'random':
                        self._secured_data[parameter] = \
                            self.randomly_generate_initial_hemodynamic_state(self._secured_data['n_node'])
                    elif tag == 'standard':
                        self._secured_data[parameter] = \
                            self.set_initial_hemodynamic_state_as_inactivated(self._secured_data['n_node'])
                    else:
                        raise ValueError('Improper tag')
                elif parameter == 'hemodynamic_parameter':
                    if tag == 'random':
                        self._secured_data[parameter] = \
                            self.randomly_generate_hemodynamic_parameters(self._secured_data['n_node'])
                    elif tag == 'standard':
                        self._secured_data[parameter] = \
                            self.get_standard_hemodynamic_parameters(self._secured_data['n_node'])
                    else:
                        raise ValueError('Improper tag')
                elif parameter == 'n_stimuli':
                    if tag == 'random':
                        self._secured_data[parameter] = \
                            self.sample_stimuli_number(self._secured_data['n_node'])
                    else:
                        raise ValueError('Improper tag')
                elif parameter == 't_delta':
                    if tag == 'random':
                        self._secured_data[parameter] = \
                            self.sample_t_delta()
                    else:
                        raise ValueError('Improper tag')
                elif parameter == 't_scan':
                    if tag == 'random':
                        self._secured_data[parameter] = \
                            self.sample_scan_time()
                    else:
                        raise ValueError('Improper tag')
                elif parameter == 'n_node':
                    if tag == 'random':
                        self._secured_data[parameter] = \
                            self.sample_node_number()
                    else:
                        raise ValueError('Improper tag')

    def set_category_three_parameter(self, parameter, value):
        pass

    def set_backup(self, key, value):
        if key == 't_scan':
            if self._secured_data['if_random_scan_time'] == False:
                self._secured_data['t_scan'] = value
            else:
                raise ValueError('t_scan cannot be set because if_random_scan_time=True')
        elif key == 't_delta':
            if self._secured_data['if_random_delta_t'] == False:
                self._secured_data['t_delta'] = value
            else:
                raise ValueError('t_delta cannot be set because if_random_delta_t=True')
        elif key == 'n_node':
            if self._secured_data['if_random_node_number'] == False:
                self._secured_data['n_node'] = value
            else:
                raise ValueError('n_node cannot be set because if_random_node_number=True')
        else:
            raise ValueError(key + ' cannot be set.')

