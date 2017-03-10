import random
import numpy as np
import scipy as sp
import scipy.ndimage
import pandas as pd
import math as mth
import scipy as sp
import scipy.stats
import os
import sys
import warnings
import collections
import matplotlib.pyplot as plt
import subprocess
import pickle
import copy

def cdr(relative_path, if_print=False):
    file_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_path + relative_path)
    if if_print:
        print('working directory is ' + os.getcwd())


def load_template(data_path):
    with open(data_path, 'rb') as f:
        template = pickle.load(f)
    return template


def load_data(data_path):
    with open(data_path, 'rb') as f:
        template = pickle.load(f)
    return template


def save_data(data_path, data):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


def mse(value_hat, value_true):
    mse = ((value_hat.flatten() - value_true.flatten()) ** 2).mean()
    return mse


def rmse(value_hat, value_true):
    error = value_hat.flatten() - value_true.flatten()
    norm_true = np.linalg.norm(value_true.flatten())
    norm_error = np.linalg.norm(error)
    rmse = norm_error / norm_true
    return rmse


def argsort2(array):
    return np.squeeze(np.dstack(np.unravel_index(np.argsort(array.ravel()), array.shape)))

def take_value(array, rc_locations):
    return [array[x[0], x[1]] for x in rc_locations]


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
                raise ValueError('Key order should be r list or dictionary')

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
            raise ValueError(key + ' is not r valid element in key order')
        else:
            return self.key_order.index(key)


def split(data, n_segment, split_dimension=0):
    """
    Split a large array data into list of segments
    :param data:
    :param split_dimension:
    :param n_segment: length of a segment
    :return:
    """
    n_total = data.shape[split_dimension]

    if n_total % n_segment == 0:
        output = np.array_split(data, int(n_total / n_segment), split_dimension)
    else:
        n_truncated = np.floor(n_total / n_segment) * n_segment
        data_truncated = data[:n_truncated]
        output = np.array_split(data_truncated, int(n_truncated / n_segment), split_dimension)
    return output

def split_advanced(data, n_segment, shift=0, split_dimension=0, n_step=None):
    """
    Split a large array data into list of segments, ignoring the beginning shift points
    :param data:
    :param n_segment:
    :param shift:
    :param split_dimension:
    :param n_step:
    :return:
    """
    n_step = n_step or n_segment
    length = data.shape[split_dimension]
    data_shifted = np.take(data, range(shift, length), split_dimension)
    n_total = length - shift
    if n_total % n_segment == 0:
        output = np.array_split(data_shifted, int(n_total / n_segment), split_dimension)
    else:
        n_truncated = np.int64(np.floor(n_total / n_segment) * n_segment)
        data_truncated = np.take(data_shifted, range(n_truncated), split_dimension)
        output = np.array_split(data_truncated, int(n_truncated / n_segment), split_dimension)
    return output

def split_data_for_initializer_graph(x_data, y_data, n_segment, shift_x_h):
    x_splits = split_advanced(x_data, n_segment)
    y_splits = split_advanced(y_data, n_segment, shift_x_h)
    n_segments = min(len(x_splits), len(y_splits))
    x_splits = x_splits[:n_segments]
    y_splits = y_splits[:n_segments]
    return [x_splits, y_splits]



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
                 h_parameter_check_statistics=None,
                 n_time_point_unit_length=None
                 ):

        self.n_node_low = n_node_low or 3
        self.n_node_high = n_node_high or 10
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

        self.u_t_low = u_t_low or 2  # in second
        self.u_t_high = u_t_high or 8  # in second

        self.h_parameter_check_statistics = h_parameter_check_statistics or 'deviation'
        self.deviation_constraint = deviation_constraint or 1
        self.hemo_parameter_keys = ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0',
                                    'x_h_coupling']
        self.hemo_parameter_mean = pd.Series([0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 100., 0.03, 25, 40.3, 0.02],
                                             self.hemo_parameter_keys)
        self.hemo_parameter_variance = pd.Series([0.0015, 0.0024, 0.015, 0.002, 0.0568, 0., 0., 0., 0., 0., 0.],
                                                 self.hemo_parameter_keys)

        self.n_time_point_unit_length = n_time_point_unit_length or 32

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
        Generate r set of matrices for neural level equation x'=Ax+\sigma(xBu)+Cu.
        For assumptions about each matrix, please refer to each individual generating function.
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :param sparse_level: sparse level of off_diagonal elements, [0, 1],
        actual non-zeros elements equal int(sparse_level * (n_node-1) * n_node)
        :return: r dictionary with elements 'A', 'B', and 'C'
        """
        sparse_level = sparse_level or self.sparse_level
        connection_matrices = {'A': self.randomly_generate_sparse_A_matrix(n_node, sparse_level),
                               'B': self.randomly_generate_B_matrix(n_node, n_stimuli),
                               'C': self.randomly_generate_C_matrix(n_stimuli, n_stimuli)}
        return connection_matrices

    def roll(self, probability):
        """
        sample r random number from [0,1), if it's smaller than or equal to given probability, return True,
        otherwise, False
        :param probability: r number from [0 ,1]
        :return: boolean
        """
        if random.random() >= probability:
            return True
        else:
            return False

    def check_transition_matrix(self, A):
        """
        Check whether all the eigenvalues of A have negative real parts
        to ensure r corresponding linear system is stable
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
        :return: r sparse A matrix
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
        :return: r list of B matrices
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
        :return: r C matrix
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
        :return: np.array, random input, size (n_time_point, n_stimuli)
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
        u = np.zeros((n_time_point, n_stimuli))

        for n_s in range(n_stimuli):
            i_current = 0
            value = 0
            while i_current < n_time_point:
                step = np.random.randint(u_n_low, u_n_high)
                value = flip(value)
                i_next = i_current + step
                if i_next >= n_time_point:
                    i_next = n_time_point
                u[i_current:i_next, n_s] = value
                i_current = i_next
        return u

    def get_impulse_u(self, n_stimuli, n_time_point):
        """
        Generate impulse stimuli.
        :param n_stimuli: the amount of stimuli
        :param n_time_point: the total sample point of stimuli
        :return: np.array, size (n_time_point, n_stimuli)
        """
        u = np.zeros((n_time_point, n_stimuli))
        u[0, :] = 1
        return u

    def get_hemodynamic_parameter_prior_distributions(self):
        """
        Get prior distribution for hemodynamic parameters (Gaussian)
        :return: r pandas.dataframe containing mean, variance, and standard deviation of each parameter
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
        :return: r dict of  pandas.dataframe, containing hemodynamic parameter distributions for all the nodes,
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
        :return: r pandas cores frame, containing hemodynamic parameters for all the nodes
        """
        return self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)['mean']

    def randomly_generate_hemodynamic_parameters(self, n_node, deviation_constraint=None):
        """
        Get random hemodynamic parameters, sampled from prior distribution.
        The sample range is constrained to mean +/- deviation_constraint * standard_deviation
        :param n_node: number of nodes (brain areas)
        :param deviation_constraint: float, used to constrain the sample range
        :return: r pandas cores frame, containing hemodynamic parameters for all the nodes,
                 and optionally, normalized standard deviation.
        """
        deviation_constraint = deviation_constraint or self.deviation_constraint

        def sample_hemodynamic_parameters(hemodynamic_parameters_mean, hemodynamic_parameters_std,
                                          deviation_constraint):
            # sample r subject from hemodynamic parameter distribution
            h_mean = hemodynamic_parameters_mean
            h_std = hemodynamic_parameters_std

            h_para = copy.deepcopy(h_mean)
            h_devi = copy.deepcopy(h_std)

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
        For each hemodynamic parameter, map how it derives from the means.
        :param hemodynamic_parameters: r pandas.dataframe, containing sampled hemodynamic parameters
        :param h_parameter_check_statistics: specify which kind of statistic to check,
               it takes value in {'deviation', 'pdf'}
        :return: r pandas.dataframe, containing checking statistics
        """
        h_parameter_check_statistics = h_parameter_check_statistics or self.h_parameter_check_statistics
        n_node = hemodynamic_parameters.shape[0]
        temp = self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)
        h_mean = temp['mean']
        h_std = temp['std']

        h_devi = copy.deepcopy(h_mean)
        h_pdf = copy.deepcopy(h_mean)
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
        :param hemodynamic_parameters: r pandas.dataframe to be checked
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
            x_h_coupling = hemodynamic_parameter.loc['region_' + str(n), 'x_h_coupling']

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

            Whx_.append(np.array([t_delta * x_h_coupling, 0, 0, 0]).reshape(4, 1))

            Wo = np.array([-(1 - epsilon) * V0, -4.3 * theta0 * E0 * V0 * TE, -epsilon * r0 * E0 * V0 * TE])
            Wo_.append(Wo)

            bh = np.array(np.asarray([t_delta * gamma, 0, 0, 0]).reshape(4, 1))
            bh_.append(bh)

            bo = V0 * (4.3 * theta0 * E0 * TE + epsilon * r0 * E0 * TE + (1 - epsilon))
            bo_.append(bo)

        return {'Whh': Whh_, 'Whx': Whx_, 'bh': bh_, 'Wo': Wo_, 'bo': bo_}

    def calculate_n_time_point(self, t_scan, t_delta):
        """
        Calculate n_time_point, n_time_point is ceil to self.n_time_point
        :param t_scan: total scan time in second
        :param t_delta: time interval for approximate differential equations in second
        :return: n_time_point
        """
        unit_length = self.n_time_point_unit_length
        para_temp = t_scan / t_delta
        n_time_point = mth.ceil(para_temp / unit_length) * unit_length
        return n_time_point


class ParameterGraph:
    def __init__(self):
        self._para_forerunner = {
            # inherited
            'initializer': [],
            'parameter_graph': [],
            'scanner': [],

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
                  't_delta',
                  'initializer'],

            'A': ['if_random_neural_parameter',
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

            'x': ["Wxx", "Wxxu", "Wxu", 'initial_x_state', 'u', 'scanner'],
            'h': ['Whh', 'Whx', 'bh', 'hemodynamic_parameter', 'x', 'scanner'],
            'y': ['Wo', 'bo', 'h', 'scanner'],
            # not necessary before estimation
            'n_backpro': [],  # number of truncated back propagation steps
            'learning_rate': [],  # used by tensorflow optimization operation
        }

        level_para = {
            'inherited': ['initializer', 'parameter_graph', 'scanner'],
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
            'level_4': ['Wxxu', 'Wxu'],
            'level_5': ['x'],
            'level_6': ['h'],
            'level_7': ['y']
        }
        level_para_order = ['inherited', 'level_0', 'level_1', 'level_2', 'level_3', 'level_4', 'level_5',
                            'level_6', 'level_7']
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
                if variable_level_map[key] <= max_forerunner_level:
                    raise ValueError(key + ' parameter graph error')
        return True

    def get_para_forerunner_mapping(self):
        return copy.deepcopy(self._para_forerunner)

    def forerunner2descendant(self, forerunner_mapping=None, key_order=None, if_complete=True):
        """
        Transfer {parameter:[forerunners]} mapping  dictionary to
        {parameter:[descendants]} mapping dictionary
        :param forerunner_mapping: parameter:forerunners mapping dictionary
        :param key_order: if an {key:order} dictionary is given, return an OderderDict, otherwise, r dict
        :param if_complete: if True, treat keys in forerunner dictionary as valid parameters.
            Parameter without r descendant will also be added to resulting dictionary
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
        return copy.deepcopy(self._level_para)

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
        :param para_forerunner: r dictionary recording _para_forerunner of each parameter
        :return: r {para:category} dictionary recording category of each parameter
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
        Find if_ flag from the prerequisites of r parameter
        :param prerequisites: r list of parameters
        :return: None, if prerequisites is empty, or there is no flag in prerequisites
                 r flag name if there is r flag in prerequisites
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
         Find if_ flag from the forerunners of r parameter
        :param para: target parameter
        :return: None, if forerunners is empty, or there is no flag in forerunners
                 r flag name if there is r flag in forerunners
        """
        self.check_valid_para(para)
        para_forerunner = self.get_para_forerunner_mapping()
        return self.abstract_flag(para_forerunner[para])

    def make_graph(self, relation_dict=None, file_name=None, rank_dict=None, rank_order=None):
        """
        Create .gv file and then use dot tool to create r diagram.
        :param relation_dict: {form:[tos]} structure, recording edges
        :param file_name: title, file name of .gv and .png file
        :param rank_dict: {rank_name:[members]} dict recording rank information
        :param rank_order: [rank_names] list recording rank order
        :return: True if it runs to the end
        """
        '''
            if relation_dict == None and file_name == None and rank_dict == None and rank_order == None:
            relation_dict = self.forerunner2descendant(self.get_para_forerunner_mapping())
            file_name = "documents/parameter_level_graph"
            rank_dict = self._level_para
            rank_order = self._level_para.key_order
        '''
        relation_dict = relation_dict or self.forerunner2descendant(self.get_para_forerunner_mapping())
        file_name = file_name or "documents/parameter_level_graph"
        rank_dict = rank_dict or self._level_para
        rank_order = rank_order or self._level_para.key_order

        with open(file_name + ".gv", 'w') as f:
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
            # however, it doesn't work at this moment
            source_file = file_name + ".gv"
            target_file = file_name + ".png"
            subprocess.run(["dot", "-Tpng", source_file, "-o", target_file], check=True)
            string = ' '.join(["dot", "-Tpng", source_file, "-o", target_file])
            print('run2 the following command in the terminal to update graph:')
            print('cd ' + os.getcwd())
            print(string)

    def get_all_para_names(self):
        """
        Get all parameter names in r DataUnit.
        :return: r list of parameter names, sorted by parameter level
        """
        para_level = self.get_para_level_mapping()
        return sorted(para_level.keys(), key=lambda key: para_level[key])

    def check_valid_para(self, para):
        """
        Check if r given parameter name is r valid one
        :param para: target name
        :return: True or raise error
        """
        valid_names = self.get_all_para_names()
        if para in valid_names:
            return True
        else:
            # print('current parameter name is ' + para)
            # string = ', '.join(valid_names)
            # print('all_para_names: ' + string)
            raise ValueError('Improper name.')


class Scanner:
    def __init__(self, snr_y=None,
                 x_max_bound=None, x_mean_bound=None,
                 x_var_low=None, x_var_high=None,
                 # x_low_frequency_energy_perscent=None,
                 h_value_low=None, h_value_high=None):
        self.snr_y = snr_y or 2
        self.x_max_bound = x_max_bound or 8
        self.x_mean_bound = x_mean_bound or 2
        self.x_var_low = x_var_low or 0.1
        self.x_var_high = x_var_high or 4
        # self.x_low_frequency_energy_perscent = x_low_frequency_energy_perscent or 0.25
        self.h_value_low = h_value_low or 0.125
        self.h_value_high = h_value_high or 8

    def scan_x(self, parameter_package):
        """
        Calculate x, namely neural activity
        :param parameter_package: r dictionary containing all needed parameters
        :return: x, np.array, (n_time_point, n_node)
        """
        Wxx = parameter_package['Wxx']
        Wxxu = parameter_package['Wxxu']
        Wxu = parameter_package['Wxu']
        initial_x_state = parameter_package['initial_x_state']
        u = parameter_package['u']

        n_node = Wxu.shape[0]
        n_stimuli = Wxu.shape[1]
        n_time_point = u.shape[0]
        x = np.zeros((n_time_point, n_node))
        x[0, :] = initial_x_state

        for i in range(1, n_time_point):
            tmp1 = np.matmul(Wxx, x[i - 1, :])
            tmp2 = [np.matmul(Wxxu[idx], x[i - 1, :] * u[i - 1, idx]) for idx in range(n_stimuli)]
            tmp2 = np.sum(np.asarray(tmp2), 0)
            tmp3 = np.matmul(Wxu, u[i - 1, :])
            x[i, :] = tmp1 + tmp2 + tmp3
        return x

    def if_proper_x(self, x):
        """
        Check if r x seems r good one in terms of some statistics which in includes:
        max absolute value and energy distribution on frequency
        :param x: neural activities, np.narray of (n_time_point, n_node)
        :return: True if x means proper; False otherwise
        """
        max_absolute_value = max(abs(x.flatten()))
        max_mean_value = np.max(abs(np.mean(x, axis=0)))
        n_time_point = x.shape[0]
        variance = np.var(x[int(n_time_point/2):, :], 0)
        min_var = min(variance)
        max_var = max(variance)

        '''
        fx = np.fft.fft(x, axis=0) / np.sqrt(x.shape[0])
        low_frequency_range = int(x.shape[0]/20)
        low_frequency_energy = 1
        energy_persentage = 1
        '''

        if max_absolute_value > self.x_max_bound:
            # print('x max_abslute_value = ' + str(max_abslute_value))
            print("Not proper x: max value")
            return False
        if max_mean_value > self.x_mean_bound:
            print("Not proper x: mean value")
            return False
        if min_var < self.x_var_low or max_var > self.x_var_high:
            # print('x min_var = ' + str(min_var))
            # print('x max_var = ' + str(max_var))
            print("Not proper x: variance")
            return False
        return True

    def if_proper_h(self, h):
        """
        Check if r set of generated h seems r reasonable one.
        Namely, f, v, q should be within [self.h_value_low, self.h_value_high] all the time
        :param h: test hemodynamic state, np.array of (n_time_point, n_node, 4)
        :return: True if x means proper; False otherwise
        """
        h_temp = h[:, :, 1:].flatten()
        min_value = np.min(h_temp)
        max_value = np.max(h_temp)
        if min_value >= self.h_value_low and max_value <= self.h_value_high:
            return True
        else:
            print("Not proper h")
            return False

    def phi_h(self, h_state_current, alpha, E0):
        """
        used to map hemodynamic states into higher dimension
        for hemodynamic states scan
        :param h_state_current:
        :param alpha:
        :param E0:
        :return:
        """
        h_state_augmented = np.zeros(7)
        h_state_augmented[0:4] = h_state_current
        h_state_augmented[4] = h_state_current[2] ** (1 / alpha)
        h_state_augmented[5] = h_state_current[3] / (h_state_current[2]) * h_state_augmented[4]
        h_state_augmented[6] = (1 - (1 - E0) ** (1 / h_state_current[1])) / (E0) * h_state_current[1]
        return h_state_augmented

    def scan_h(self, parameter_package):
        """
        Calculate h, namely hemodynamic response
        :param parameter_package: r dictionary containing all needed parameters
        :return: np.array of size (n_time_point, n_node, 4)
        """
        hemodynamic_parameters = parameter_package['hemodynamic_parameter']
        Whh = parameter_package['Whh']
        bh = parameter_package['bh']
        Whx = parameter_package['Whx']
        initial_h_state = parameter_package['initial_h_state']
        x = parameter_package['x']

        n_time_point = x.shape[0]
        n_node = x.shape[1]
        eps = np.finfo(float).eps
        h = np.ones((n_time_point, n_node, 4))
        h[:, :, 0] = 0
        h[0, :, :] = initial_h_state

        for t in range(1, n_time_point):
            for n in range(0, n_node):
                alpha = hemodynamic_parameters.loc['region_' + str(n), 'alpha']
                E0 = hemodynamic_parameters.loc['region_' + str(n), 'E0']
                h_temp = (np.matmul(Whh[n], self.phi_h(h[t - 1, n, :], alpha, E0)).reshape(4, 1)
                          + Whx[n] * x[t - 1, n] + bh[n]).reshape(4)
                '''
                # do not use exp trick to protect f, v, q to be possitive
                # it induces error in our approximation
                # s doesn't need protection
                h[t, n, 0] = h_temp[0]
                # avoid f, v, q run2 into non-positive value
                fvq_tm1 = h[t - 1, n, 1:]
                fvq_temp = h_temp[1:]
                fvq_delta = fvq_temp - fvq_tm1
                fvq_t = fvq_tm1 * np.exp(fvq_delta / fvq_tm1)
                fvq_t = np.array([value if value > eps else eps for value in list(fvq_t)])
                h[t, n, 1:] = fvq_t
                '''
                h[t, n, :] = h_temp
        return h

    def phi_o(self, h_state_current):
        """
        used to map hemodynamic states into higher dimension
        for fMRI ys
        :param h_state_current:
        :return:
        """
        o_state_augmented = np.zeros(3)
        o_state_augmented[0:2] = h_state_current[2:4]
        o_state_augmented[2] = o_state_augmented[1] / o_state_augmented[0]
        return o_state_augmented

    def scan_y(self, parameter_package):
        """
        Calculate value, namely observable functional signal
        :param parameter_package: r dictionary containing all needed parameters
        :return: np.array of size (n_time_point, n_node)
        """

        Wo = parameter_package['Wo']
        bo = parameter_package['bo']
        h = parameter_package['h']
        n_time_point = h.shape[0]
        n_node = h.shape[1]
        y = np.zeros((n_time_point, n_node))

        for t in range(0, n_time_point):
            for n in range(0, n_node):
                y[t, n] = np.matmul(Wo[n], self.phi_o(h[t, n, :])) + bo[n]
        return y


class DataUnit(Initialization, ParameterGraph, Scanner):
    """
    This class is used to ensure consistence and integrity of all cores, but that takes r lot of efforts, so currently,
    it's used in r unsecured manner. Namely, DataUnit inherits dict and it key/value pair can be changed without other
    constraints, which means they might not be consistent.
    A internal dictionary, _secured_data is r dictionary should only manipulated by internal methods. It's not
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
        Scanner.__init__(self)
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
        self._secured_data['parameter_graph'] = self
        self._secured_data['scanner'] = self

        # when do auto cores generating, following the order below
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
                              'Wo', 'bo',
                              'x', 'h', 'y']

        # record cores which should be kept before auto complement
        self._locked_data = {}
        self.lock_current_data()

    def get_assign_order(self, start_category=1):
        para_cate = self.get_para_category_mapping()
        assign_order = [para for para in self._assign_order if para_cate[para] >= start_category]
        return assign_order

    def lock_current_data(self):
        self._locked_data = copy.deepcopy(self._secured_data)

    def refresh_data(self):
        self._secured_data = copy.deepcopy(self._locked_data)

    def get_locked_data(self):
        return copy.deepcopy(self._locked_data)

    def set(self, para, value):
        """
        Set value to r parameter.
        A parameter can only be directly assigned r number if it is of category one or two (see ParameterGraph)
        When r parameter is assigned r number, its compatibility should be checked with existing parameters.
        But it takes r lot of time to implement.
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

    def get(self, para_name):
        """
        Get r parameter from _secured_data
        :param para_name:
        :return:
        """
        return copy.deepcopy(self._secured_data[para_name])

    def if_has_no_assigned_descendant(self, para):
        """
        Check the descendants of r para, if any of them has r value, return False
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
        Check the descendants of r para, if any of them has r value, raise error
        :param para: target parameter
        :return: If no descendant has had value, return True; otherwise, error.
        """
        flag = self.if_has_no_assigned_descendant(para)
        if flag:
            return True
        else:
            raise ValueError(para + ' has descendant with value')

    def if_has_value(self, para, search_target_dictionary=None):
        """
        Check whether if r parameter has been assigned r value in search_target_dictionary
        :param para: target para
        :param search_target_dictionary:
        :return: True if para has r value; otherwise False
        """
        search_target_dictionary = search_target_dictionary or self._secured_data
        self.check_valid_para(para)
        if para in search_target_dictionary.keys():
            return True
        else:
            return False

    def check_all_have_values(self, parameters, search_target_dictionary=None):
        """
        Check whether all parameters in the list parameters have true_values in search_target_dictionary
        :param parameters: r list of target parameters
        :param search_target_dictionary: where to search true_values
        :return: True if all parameters have true_values; otherwise raise error
        """
        search_target_dictionary = search_target_dictionary or self._secured_data
        flags = [self.if_has_value(para, search_target_dictionary) for para in parameters]
        if False in flags:
            parameters = [parameters[index] for index, value in enumerate(flags) if value is False]
            string = ', '.join(parameters)
            raise ValueError(string + " has (have) not been specified.")
        else:
            return True

    def complete_data_unit(self, start_categorty=1, if_check_property=True, if_show_message=False):
        """
        Generate missing parameters
        :param start_categorty: if it starts from category n, category n parameters must have had true_values
        :param if_show_message: boolean, whether show completing message
        :return:
        """
        # check category one parameters
        cate_para = self.get_category_para_mapping()
        parameters_needed = cate_para[start_categorty]
        self.check_all_have_values(parameters_needed)
        assign_order = self.get_assign_order(start_categorty)
        if if_check_property:
            trail_count = 1
            print("Trail NO." + str(trail_count))
            self.lock_current_data()
            self._simple_complete(assign_order, if_show_message)
            while not self.if_proper_x(self._secured_data['x']) or not self.if_proper_h(self._secured_data['h']):
                self.refresh_data()
                trail_count += 1
                print("Trail NO." + str(trail_count))
                self._simple_complete(assign_order, if_show_message)
        else:
            self._simple_complete(assign_order, if_show_message)

    def _simple_complete(self, assign_order, if_show_message=False):
        """
        complete dataUnit without check quality
        :param assign_order: r list indicating parameter assign order
        :return:
        """
        for para in assign_order:
            # self.check_has_no_assigned_descendant(para)
            if not self.if_has_value(para):
                flag_name = self.get_flag_name(para)
                self._set(para, flag_name, if_show_message)
            else:
                if if_show_message:
                    print(para + ' has value:')
                    print(self._secured_data[para])

    def _set(self, para, flag_name, if_show_message=False):
        """
        Used by auto cores generating process.
        Assume all constraints have been checked, and value assignment is allowed.
        Generate value for para following flag information.
        :param para: target para
        :param flag_name: if_random flag
        :return:
        """

        def show(para_name, flag_name, flag_value, method):
            if if_show_message:
                if flag_name is None:
                    flag_name = 'None_flag'
                message = flag_name + ' is ' + str(flag_value) + ', ' + para_name + ' is set by ' + method
                print(message)

        def raise_error(para_name, flag_name, flag_value, error):
            message = flag_name + ' is ' + str(flag_value) + ', ' + para_name + ' ' + error
            raise ValueError(message)

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
            show(para, flag_name, flag_value, 'calculate_n_time_point')
            self._secured_data[para] = self.calculate_n_time_point(self._secured_data['t_scan'],
                                                                   self._secured_data['t_delta'])
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
                show(para, flag_name, flag_value, 'randomly_generate_hemodynamic_parameters')
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
            show(para, flag_name, flag_value, 'calculate_dcm_rnn_x_matrices')
            para_temp = self.calculate_dcm_rnn_x_matrices(self._secured_data['A'],
                                                          self._secured_data['B'],
                                                          self._secured_data['C'],
                                                          self._secured_data['t_delta'])
            self._secured_data[para] = para_temp[para]
        elif para in ['Whh', 'Whx', 'bh', 'Wo', 'bo']:
            assert flag_name is None
            show(para, flag_name, flag_value, 'calculate_dcm_rnn_h_matrices')
            para_temp = self.calculate_dcm_rnn_h_matrices(self._secured_data['hemodynamic_parameter'],
                                                          self._secured_data['t_delta'])
            self._secured_data[para] = para_temp[para]
        elif para is 'x':
            assert flag_name is None
            show(para, flag_name, flag_value, 'scan_x')
            parameter_package = self.collect_parameter_for_x_scan()
            self._secured_data[para] = self.scan_x(parameter_package)
        elif para is 'h':
            assert flag_name is None
            show(para, flag_name, flag_value, 'scan_h')
            parameter_package = self.collect_parameter_for_h_scan()
            self._secured_data[para] = self.scan_h(parameter_package)
        elif para is 'y':
            assert flag_name is None
            show(para, flag_name, flag_value, 'scan_y')
            parameter_package = self.collect_parameter_for_y_scan()
            self._secured_data[para] = self.scan_y(parameter_package)

    def get_dcm_rnn_x_matrices(self):
        """
        Return connection matrices of neural equation in DCM_RNN
        :return: [Wxx, Wxxu, Wxu]
        """
        assert 'Wxx' in self._secured_data.keys()
        assert 'Wxxu' in self._secured_data.keys()
        assert 'Wxu' in self._secured_data.keys()
        return [self._secured_data['Wxx'],
                self._secured_data['Wxxu'],
                self._secured_data['Wxu']]

    def check_forerunners(self, para):
        """
        Check if all forerunners of r para have been specified
        :param para: target para
        :return: True or raise error
        """
        para_forerunners = self.get_para_forerunner_mapping()
        forerunner_checks = [para in self._secured_data.keys()
                             for para in para_forerunners[para]]
        if False in forerunner_checks:
            not_satisfied_forerunners = [para_forerunners[para][idx]
                                         for idx, val in enumerate(forerunner_checks) if not val]
            string = ', '.join(not_satisfied_forerunners)
            raise ValueError(para + " cannot be assigned because the following prerequisites have not be "
                             + "assigned: " + string)
        else:
            return True

    def collect_parameter_for_x_scan(self):
        return {'Wxx': self._secured_data['Wxx'],
                'Wxxu': self._secured_data['Wxxu'],
                'Wxu': self._secured_data['Wxu'],
                'initial_x_state': self._secured_data['initial_x_state'],
                'u': self._secured_data['u']}

    def collect_parameter_for_h_scan(self):
        return {'hemodynamic_parameter': self._secured_data['hemodynamic_parameter'],
                'Whh': self._secured_data['Whh'],
                'bh': self._secured_data['bh'],
                'Whx': self._secured_data['Whx'],
                'initial_h_state': self._secured_data['initial_h_state'],
                'x': self._secured_data['x']}

    def collect_parameter_for_y_scan(self):
        return {'Wo': self._secured_data['Wo'],
                'bo': self._secured_data['bo'],
                'h': self._secured_data['h']}

    def lock_current_data(self):
        """
        Copy cores in _secured_data into _locked_data
        :return:
        """
        self._locked_data = copy.deepcopy(self._secured_data)

    def collect_parameter_core(self):
        """
        :return: Return the initial setting and core paremeters, with which the DCM model can be reproduced.
                 Namely, the category one and two parameters
        """
        para_core = dict()
        cate_para = dict(self.get_category_para_mapping())
        core_para = cate_para[1] + cate_para[2]
        self.check_all_have_values(core_para)
        for para in core_para:
            para_core[para] = self._secured_data[para]
        return para_core

    def load_parameter_core(self, parameter_core):
        """
        Load the given parameter_core
        :param parameter_core: r dictionary having core parameters need to reproduced original dateUnit
        :return:
        """
        cate_para = dict(self.get_category_para_mapping())
        core_para = cate_para[1] + cate_para[2]
        self.check_all_have_values(core_para, parameter_core)
        self._secured_data = copy.deepcopy(parameter_core)

    def recover_data_unit(self):
        """
        :return:
        """
        self.complete_data_unit(start_categorty=2, if_check_property=False)

    def plot(self, para_name, if_new_figure=True):
        para_value = self._secured_data[para_name]
        n_node = self._secured_data['n_node']
        x_axis = np.arange(self._secured_data['n_time_point']) * self._secured_data['t_delta']
        if para_name in ['x', 'y']:
            plt.clf()
            for n in range(n_node):
                value_temp = para_value[:, n]
                plt.subplot(n_node, 1, n + 1)
                plt.plot(x_axis, value_temp)
        elif para_name is 'h':
            if if_new_figure:
                plt.figure()
            for n in range(n_node):
                value_temp = para_value[:, n, :].squeeze()
                plt.subplot(n_node, 1, n + 1)
                plt.plot(x_axis, value_temp)

    def map(self, source_para_name, source_para_values, target_para_name):
        """
        After loading r parameter core, use different true_values of source parameter and ys target parameter,
        to map the influence of source parameter on the target parameter. Supported source_para [t_delta]
        :param source_para_name:
        :param source_para_values: r list of true_values source parameters
        :param target_para_name:
        :return: r list of target parameters
        """
        output = []
        self.lock_current_data()
        if source_para_name == 't_delta':
            t_delta_original = self._secured_data['t_delta']
            for value in source_para_values:
                self.refresh_data()
                self._secured_data[source_para_name] = value

                # all source_para_name's descendants need to be modified accordingly
                # for t_delta, u must be manually modified
                u = self._secured_data['u']
                factor = np.ones(u.ndim)
                factor[0] = t_delta_original/value
                u_rescaled = scipy.ndimage.zoom(u, factor, order=0, mode='nearest')
                self._secured_data['u'] = u_rescaled

                self.recover_data_unit()
                output.append(self._secured_data[target_para_name])
            self.refresh_data()
            return output
        elif source_para_name in ['A', 'B', 'C']:
            for value in source_para_values:
                self.refresh_data()
                self._secured_data[source_para_name] = value
                self.recover_data_unit()
                output.append(self._secured_data[target_para_name])
                print(self.get('Wxx'))
            self.refresh_data()
            return output
        elif source_para_name in ['effictive_connection']:
            for value in source_para_values:
                self.refresh_data()
                self._secured_data['A'] = value['A']
                self._secured_data['B'] = value['B']
                self._secured_data['C'] = value['C']
                self.recover_data_unit()
                output.append(self._secured_data[target_para_name])
            self.refresh_data()
            return output
        else:
            print(source_para_name + ' is not supported yet.')

    def resample(self, states, target_shape):
        """
        Resample states to same shape for comparison.
        :param states: r list of states
        :param target_shape: desired shape
        :return: r list of states of the same time point.
        """
        states_resampled = []
        for state in states:
            factor = np.array(target_shape)/np.array(state.shape)
            state_resampled = scipy.ndimage.zoom(state, factor, order=0, mode='nearest')
            states_resampled.append(state_resampled)
        return states_resampled

    def compare(self, arrays, ground_truth):
        """
        Compare and calculate rMSE of each in arrays against ground_truth
        :param arrays: r list of arrays of the same shape
        :param ground_truth: r array, used as ground truth
        :return: r list of rMSE
        """
        rMSE = []
        ground_truth = ground_truth.flatten()
        norm_ground_truth = np.linalg.norm(ground_truth)
        for array in arrays:
            error = array.flatten() - ground_truth
            norm_error = np.linalg.norm(error)
            rMSE.append(norm_error/norm_ground_truth)
        return rMSE

    @property
    def secured_data(self):
        return self._secured_data

