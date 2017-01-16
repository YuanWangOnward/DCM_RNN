import unittest
import numpy as np
from DCM_RNN import toolboxes
import random
import pandas as pd
import scipy as sp
import scipy.stats


class Initialization_tests(unittest.TestCase):
    def setUp(self):
        self.utl = toolboxes.Initialization()

    def tearDown(self):
        del self.utl

    def test_randomly_generate_A_matrix(self):
        if_print = False
        n_test = 10
        for n in range(n_test):
            n_node = random.randint(3, 10)
            sparse_level = random.random()
            A = self.utl.randomly_generate_A_matrix(n_node, sparse_level)
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

    def test_get_expanded_hemodynamic_parameter_prior_distributions(self):
        if_print = False
        n_node = random.randint(3, 10)
        temp = self.utl.get_expanded_hemodynamic_parameter_prior_distributions(n_node)
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

    def test_sample_node_number(self):
        n_test = 10
        for n in range(n_test):
            n_node = self.utl.sample_node_number()
            self.assertTrue((n_node >= self.utl.n_node_low))
            self.assertTrue((n_node < self.utl.n_node_high))

    def test_sample_stimuli_number(self):
        n_test = 10
        for n in range(n_test):
            n_node = self.utl.sample_node_number()
            stimuli_node_ratio = self.utl.stimuli_node_ratio
            n_stimuli = self.utl.sample_stimuli_number(n_node, stimuli_node_ratio)
            self.assertTrue(n_stimuli >= 1)
            self.assertTrue(n_stimuli <= int(stimuli_node_ratio * n_node))

    def test_sample_t_delta(self):
        n_test = 10
        for n in range(n_test):
            t_delta = self.utl.sample_t_delta()
            self.assertTrue(t_delta >= self.utl.t_delta_low)
            self.assertTrue(t_delta < self.utl.t_delta_high)

    def test_sample_scan_time(self):
        n_test = 10
        for n in range(n_test):
            scan_time = self.utl.sample_scan_time()
            self.assertTrue(scan_time >= self.utl.scan_time_low)
            self.assertTrue(scan_time < self.utl.scan_time_high)


class ParameterGraph_tests(unittest.TestCase):
    def setUp(self):
        self.pg = toolboxes.ParameterGraph()

    def tearDown(self):
        del self.pg

    def test_generate_gv_file(self):
        if_update_graph = True
        if if_update_graph:
            self.pg.generate_gv_file()

    def test_check_parameter_relation(self):
        self.assertTrue(self.pg.check_parameter_relation())

class DataUnit_tests(unittest.TestCase):
    def setUp(self):
        self.du = toolboxes.DataUnit()

    def tearDown(self):
        del self.du

    def test_set(self):
        self.du.set('t_delta', 0.2)
        self.assertTrue((self.du._secured_data['t_delta']) == 0.2)
        self.du.set('n_node', 3)
        self.assertTrue((self.du._secured_data['n_node']) == 3)
        self.du.set('t_scan', 400)
        self.assertTrue((self.du._secured_data['t_scan']) == 400)
        with self.assertRaises(ValueError):
            self.du.set('u', 400)
        with self.assertRaises(ValueError):
            self.du.set('A', 400)
        self.du._secured_data['if_random_delta_t'] = True
        with self.assertRaises(ValueError):
            self.du.set('t_delta', 0.2)
        self.du._secured_data['if_random_node_number'] = True
        with self.assertRaises(ValueError):
            self.du.set('n_node', 3)
        self.du._secured_data['if_random_scan_time'] = True
        with self.assertRaises(ValueError):
            self.du.set('t_scan', 400)






if __name__ == '__main__':
    unittest.main()
