import random
import numpy as np
import pandas as pd

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

    def get_stardard_hemodynamic_parameters(self, n_region):
        hemodynamic_parameters_mean = pd.DataFrame()
        hemo_parameter_key_list = ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0']
        hemo_parameter_mean_list = [0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 100., 0.03, 25, 40.3]
        for idx, key in enumerate(hemo_parameter_key_list):
            tmp = [hemo_parameter_mean_list[idx] for _ in range(n_region)]
            tmp = pd.Series(tmp, index=['region_' + str(i) for i in range(n_region)])
            hemodynamic_parameters_mean[key] = tmp
        return hemodynamic_parameters_mean


