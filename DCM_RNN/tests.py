import unittest
import numpy as np
from DCM_RNN import utilities
import random
import pandas as pd
import scipy as sp
import scipy.stats


class Utilities_tests(unittest.TestCase):
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

    def test_randomly_generate_sparse_A_matrix(self):
        if_print = False
        n_test = 10
        for n in range(n_test):
            n_node = random.randint(3, 10)
            sparse_level = random.random()
            A = self.utl.randomly_generate_sparse_A_matrix(n_node, sparse_level)
            w, v = np.linalg.eig(A)
            if if_print:
                print('created A matrix:')
                print(A)
                print('max real part = ' + str(max(w.real)))
            # eigenvalues must be negative
            self.assertTrue(max(w) < 0, 'eigenvalues of A are not all negative')
            # sparsity check
            self.assertTrue(sum(abs(A.flatten()) > 0) == n_node + int(sparse_level * (n_node - 1) * n_node),
                            'A matrix sparsity level error')

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

    def test_get_hemodynamic_parameter_prior_distributions(self):
        if_print = False
        temp = self.utl.get_hemodynamic_parameter_prior_distributions()
        if if_print:
            print(temp)

    def test_get_hemodynamic_parameter_prior_distributions_as_dataframe(self):
        if_print = False
        n_node = random.randint(3, 10)
        temp = self.utl.get_hemodynamic_parameter_prior_distributions_as_dataframe(n_node)
        if if_print:
            print(temp)

    def test_get_standard_hemodynamic_parameters(self):
        if_print = False
        n_node = random.randint(3, 10)
        temp = self.utl.get_standard_hemodynamic_parameters(n_node)
        if if_print:
            print(temp)

    def test_randomly_generate_hemodynamic_parameters(self):
        if_print = False
        n_node = random.randint(3, 10)
        output = self.utl.randomly_generate_hemodynamic_parameters(n_node)
        if if_print:
            print(output)

    def test_check_hemodynamic_parameters(self):
        if_print = False
        n_node = random.randint(3, 10)
        h_para = self.utl.randomly_generate_hemodynamic_parameters(n_node)
        output = self.utl.check_hemodynamic_parameters(h_para)
        if if_print:
            print(output)

    def test_if_proper_hemodynamic_parameters(self):
        n_test = 10
        for n in range(n_test):
            n_node = random.randint(3, 10)
            deviation_constrain = random.uniform(1, 3)
            h_para = self.utl.randomly_generate_hemodynamic_parameters(n_node, deviation_constrain)
            temp = self.utl.if_proper_hemodynamic_parameters(h_para, deviation_constrain)
            self.assertTrue(temp, 'The given hemodynamic parameters are not qualified by given deviation_constraint')


if __name__ == '__main__':
    unittest.main()
