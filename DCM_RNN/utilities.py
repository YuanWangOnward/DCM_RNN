import random
import numpy as np
import pandas as pd
import math as mth
import scipy as sp
import scipy.stats


class Utilities:
    def randomly_initialize_connection_matrices(self, n_node, n_stimuli, sparse_level=0.5):
        """
        Generate a set of matrices for neural level equation x'=Ax+\sigma(xBu)+Cu.
        For assumptions about each matrix, please refer to each individual generating function.
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :param sparse_level: sparse level of off_diagonal elements, [0, 1],
        actual non-zeros elements equal int(sparse_level * (n_node-1) * n_node)
        :return: a dictionary with elements 'A', 'B', and 'C'
        """
        connection_matrices = {'A': self.randomly_generate_sparse_A_matrix(n_node, sparse_level),
                               'B': self.randomly_generate_B_matrix(n_node, n_stimuli),
                               'C': self.randomly_generate_C_matrix(n_stimuli, n_stimuli)}
        return connection_matrices

    def roll(self, probability):
        """
        sample a random number from [0,1), if it's larger than or equal to given probability, return True,
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

    def randomly_generate_A_matrix(self, n_node):
        """
        Generate the A matrix for neural level equation x'=Ax+\sigma(xBu)+Cu.
        Eigenvalues of A must to negative to ensure the system is stable.
        Assumption:
        Diagonal elements are negative meaning self-feedback is negative.
        Strength range of diagonal elements is (-0.5, 0].
        Strength range of off-diagonal elements is [-0.45, 0.45)
        :param n_node: number of nodes (brain areas)
        :return: an A matrix
        """

        def get_a_matrix(n_node):
            A = (np.random.random((n_node, n_node)) - 0.5) * 0.9
            A = A * (1 - np.identity(n_node)) - np.diag(np.random.random(n_node) * 0.5)
            return A

        A = get_a_matrix(n_node)
        count = 0
        max_count = 5000
        while not self.check_transition_matrix(A):
            A = get_a_matrix(n_node)
            count += 1
            if count > max_count:
                raise ValueError('Can not generate qualified A matrix with max trail number!')
        return A

    def randomly_generate_sparse_A_matrix(self, n_node, sparse_level):
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

        def get_a_sparse_matrix(n_node, sparse_level):
            A = (np.random.random((n_node, n_node)) - 0.5) * 0.9
            A = (A * (1 - np.identity(n_node))).flatten()
            sorted_indexes = np.argsort(np.abs(A), None)  # ascending order
            kept_number = int(sparse_level * (n_node - 1) * n_node)
            A[sorted_indexes[list(range(0, len(A)-kept_number))]] = 0
            A = A.reshape([n_node, n_node])
            A = A * (1 - np.identity(n_node)) - np.diag(np.random.random(n_node) * 0.5)
            return A

        if sparse_level > 1 or sparse_level < 0:
            raise ValueError('Imporper sparse_level, it should be [0,1]')
        A = get_a_sparse_matrix(n_node, sparse_level)
        count = 0
        max_count = 5000
        while not self.check_transition_matrix(A):
            A = get_a_sparse_matrix(n_node, sparse_level)
            count += 1
            if count > max_count:
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
            if self.roll(0.5):
                index_temp = indexes.pop()
                row_index = int(index_temp / n_node)
                column_index = index_temp % n_node
                sign = -1 if self.roll(0.5) else 1
                B_current[row_index, column_index] = sign * random.uniform(0.2, 0.5)
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
            C[row_index, culumn_index] = np.random.random(1) * 0.5 + 0.5
        return C

    def randomly_initialize_hemodynamic_parameters(self):
        pass

    def get_hemodynamic_parameter_prior_distributions(self):
        """
        Get prior distribution for hemodynamic parameters (Gaussian)
        :return: a dictionary containing mean, variance, and standard deviation of each parameter
        """
        hemo_parameter_key_list = ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0']
        hemo_parameter_mean_list = [0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 100., 0.03, 25, 40.3]
        hemo_parameter_variance_list = [0.0015, 0.0024, 0.015, 0.002, 0.0568, 0., 0., 0., 0., 0.]
        prior_distribution = {}
        for n in range(len(hemo_parameter_key_list)):
            temp = {}
            temp['mean'] = hemo_parameter_mean_list[n]
            temp['variance'] = hemo_parameter_variance_list[n]
            temp['std'] = np.sqrt(hemo_parameter_variance_list[n])
            prior_distribution[hemo_parameter_key_list[n]] = temp
        prior_distribution['ordered_keys'] = hemo_parameter_key_list
        return prior_distribution

    def get_hemodynamic_parameter_prior_distributions_as_dataframe(self, n_node):
        """
        Repeat hemodynamic parameter prior distributions for each node and structure the results into pandas.dataframe
        :param n_node: number of nodes (brain areas)
        :return: a list of  pandas dataframes, containing hemodynamic parameter distributions for all the nodes,
                 distribution parameters include mean and standard deviation.
        """
        distributions = self.get_hemodynamic_parameter_prior_distributions()
        hemodynamic_parameters_mean = pd.DataFrame()
        hemodynamic_parameters_std = pd.DataFrame()
        for idx, key in enumerate(distributions['ordered_keys']):
            # add mean
            tmp = [distributions[key]['mean'] for _ in range(n_node)]
            tmp = pd.Series(tmp, index=['region_' + str(i) for i in range(n_node)])
            hemodynamic_parameters_mean[key] = tmp
            # add variance
            tmp = [distributions[key]['std'] for _ in range(n_node)]
            tmp = pd.Series(tmp, index=['region_' + str(i) for i in range(n_node)])
            hemodynamic_parameters_std[key] = tmp
        return {'mean': hemodynamic_parameters_mean, 'std': hemodynamic_parameters_std}

    def get_standard_hemodynamic_parameters(self, n_node):
        """
        Get standard hemodynamic parameters, namely, the means of prior distribution of hemodynamic parameters
        :param n_node: number of nodes (brain areas)
        :return: a pandas data frame, containing hemodynamic parameters for all the nodes
        """
        return self.get_hemodynamic_parameter_prior_distributions_as_dataframe(n_node)['mean']

    def randomly_generate_hemodynamic_parameters(self, n_node, deviation_constraint=1):
        """
        Get random hemodynamic parameters, sampled from prior distribution.
        The sample range is constrained to mean +/- deviation_constraint * standard_deviation
        :param n_node: number of nodes (brain areas)
        :param deviation_constraint: float, used to constrain the sample range
        :return: a pandas data frame, containing hemodynamic parameters for all the nodes,
                 and optionally, normalized standard deviation.
        """
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
        temp = self.get_hemodynamic_parameter_prior_distributions_as_dataframe(n_node)
        hemodynamic_parameters_mean = temp['mean']
        hemodynamic_parameters_variance = temp['std']
        h_para = sample_hemodynamic_parameters(hemodynamic_parameters_mean,
                                                       hemodynamic_parameters_variance,
                                                       deviation_constraint)
        return h_para

    def check_hemodynamic_parameters(self, hemodynamic_parameters, check_statistics='deviation'):
        """
        For each hemodynamic parameter, check whether it is sampled properly.
        :param hemodynamic_parameters: a pandas.dataframe, containing sampled hemodynamic parameters
        :param check_statistics: specify which kind of statistic to check,
               it takes value in {'deviation', 'pdf'}
        :return: a pandas.dataframe, containing checking statistics
        """
        n_node = hemodynamic_parameters.shape[0]
        temp = self.get_hemodynamic_parameter_prior_distributions_as_dataframe(n_node)
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

        if check_statistics == 'deviation':
            return h_devi
        elif check_statistics == 'pdf':
            return h_pdf
        else:
            raise ValueError('Improper check_statistics value!')

    def if_proper_hemodynamic_parameters(self, hemodynamic_parameters, deviation_constraint=1):
        """
        Check if given hemodynamic parameters are within deviation constraint.
        If yes, return True; otherwise, throw error
        :param hemodynamic_parameters: a pandas.dataframe to be checked
        :param deviation_constraint: target deviation constraint, in normalized deviation,
        say if deviation_constraint=1, parameter with +/- one normalized deviation is acceptable
        :return: True or Error
        """
        h_stat = self.check_hemodynamic_parameters(hemodynamic_parameters, check_statistics='deviation')
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

    def sample_initial_neural_state(self, n_node, low=0, high=0.4):
        """
        Sample initial neural state. The distribution is set by experience.
        :param n_node: number of nodes (brain areas)
        :param low: lower bound of uniform distribution
        :param high: higher bound of uniform distribution
        :return: sampled initial neural state
        """
        return np.random.uniform(low, high, size=n_node)

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
        h_state_initial[:, 0] = np.random.uniform(low=-0.3, high=0.3, size=n_node)
        h_state_initial[:, 1] = np.random.uniform(low=0.8, high=1.4, size=n_node)
        h_state_initial[:, 2] = np.random.uniform(low=0.8, high=1.4, size=n_node)
        h_state_initial[:, 3] = np.random.uniform(low=0.6, high=1.2, size=n_node)
        return h_state_initial





