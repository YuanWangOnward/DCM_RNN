import random
import numpy as np


class Utilities:
    def randomly_initilize_connection_matrices(self, n_node, n_stimuli):
        pass

    def randomly_generate_A_matrix(self, n_node):
        '''
        S = np.diag(random.sample(set(np.arange(1, 1000)/1000), n_node))
        P = np.random.random((n_node, n_node))
        U, s, V = np.linalg.svd(P)
        P = np.dot(np.dot(U, np.identity(n_node)), V)
        # P = np.triu(P, 1) + np.identity(n_node)
        P_inverse = np.linalg.inv(P)
        return np.dot(P, np.dot(S, P_inverse))
        #return np.dot(U, np.dot(S, np.matrix.transpose(U)))
        '''

        def get_a_matrix(n_node):
            A = (np.random.random((n_node, n_node)) - 0.5) * 0.9
            A = A * (1 - np.identity(n_node)) - np.diag(np.random.random(n_node) * 0.5)
            return A

        def check_matrix(A):
            # A linear is stable if all the eigenvalues of A have negative real parts
            w, v = np.linalg.eig(A)
            if max(w.real) < 0:
                return True
            else:
                return False

        A = get_a_matrix(n_node)
        count = 0
        while not check_matrix(A):
            A = get_a_matrix(n_node)
            count += 1
            if count > 2000:
                raise ValueError('Can not generate qualified A matrix with max trail number!')
        return A
