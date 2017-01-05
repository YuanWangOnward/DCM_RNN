import random
import numpy as np


class Utilities:
    def randomly_initilize_connection_matrices(self, n_node, n_stimuli):
        pass

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
        Generate an A matrix for x'=Ax+Bu.
        Eigenvalues of A must to negative to ensure the system is stable.
        :param n_node: number of nodes
        :return: an A matrix
        """
        def get_a_matrix(n_node):
            A = (np.random.random((n_node, n_node)) - 0.5) * 0.9
            A = A * (1 - np.identity(n_node)) - np.diag(np.random.random(n_node) * 0.5)
            return A

        A = get_a_matrix(n_node)
        count = 0
        while not self.check_transition_matrix(A):
            A = get_a_matrix(n_node)
            count += 1
            if count > 2000:
                raise ValueError('Can not generate qualified A matrix with max trail number!')
        return A
