import unittest
import numpy as np
from DCM_RNN import utilities
import random


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.utl = utilities.Utilities()

    def tearDown(self):
        del self.utl

    def test_randomly_generate_A_matrix(self):
        if_print = False
        n_test = 10
        for n in range(n_test):
            size = random.randint(3, 10)
            A = self.utl.randomly_generate_A_matrix(size)
            w, v = np.linalg.eig(A)
            if if_print:
                print('created A matrix:')
                print(A)
                print('max real part = ' + str(max(w.real)))
            self.assertTrue(max(w) < 0, 'eigenvalues of A are not all negtive')

    def test_randomly_generate_C_matrix(self):
        if_print = False
        n_test = 10
        for n in range(n_test):
            n_node = random.randint(3, 10)
            n_stimuli = random.sample(range(1, n_node), 1)[0]
            C = self.utl.randomly_generate_C_matrix(n_node, n_stimuli)
            max_value = max(C.flatten())
            min_value = min(C[C > 0])
            if if_print:
                print(C)
                print(max_value)
                print(min_value)
            self.assertTrue(max_value < 1, 'C matrix max value error')
            self.assertTrue(min_value >= 0.5, 'C matrix min value error')
            for column in C.transpose():
                temp = sum(abs(column) > 0)
                self.assertTrue(temp == 1, 'assignment number error')

    def test_randomly_generate_B_matrix(self):
        n_test = 10
        for n in range(n_test):
            n_node = random.randint(3, 10)
            n_stimuli = random.sample(range(1, n_node), 1)[0]
            B = self.utl.randomly_generate_B_matrix(n_node, n_stimuli)
            nonzero_indexes = []
            nonzero_values = []
            for B_current in B:
                nonzero_indexes_current = [index for index,value in enumerate(B_current.flatten()) if value != 0]
                nonzero_values_current = [value for value in B_current.flatten() if value != 0]
                nonzero_indexes.extend(nonzero_indexes_current)
                nonzero_values.extend(nonzero_values_current)
            if (len(nonzero_indexes) > 0):
                self.assertTrue(len(nonzero_indexes) == len(set(nonzero_indexes)))
                self.assertTrue(np.max(np.abs(nonzero_values))<=0.5)
                self.assertTrue(np.min(np.abs(nonzero_values)) >= 0.2)

if __name__ == '__main__':
    unittest.main()
