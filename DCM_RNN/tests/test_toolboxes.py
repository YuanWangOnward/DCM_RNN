import unittest
import numpy as np
from DCM_RNN import toolboxes
import random
from DCM_RNN.toolboxes import OrderedDict
import pandas as pd
import scipy as sp
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import math as mth


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
                nonzero_indexes_current = [index for index, value in enumerate(B_current.flatten()) if value != 0]
                nonzero_values_current = [value for value in B_current.flatten() if value != 0]
                nonzero_indexes.extend(nonzero_indexes_current)
                nonzero_values.extend(nonzero_values_current)
            if (len(nonzero_indexes) > 0):
                self.assertTrue(len(nonzero_indexes) == len(set(nonzero_indexes)))
                self.assertTrue(np.max(np.abs(nonzero_values)) <= 0.5)
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
        output = self.utl.evaluate_hemodynamic_parameters(h_para)
        if if_print:
            print(output)

    def test_if_proper_hemodynamic_parameters(self):
        n_test = 10
        for n in range(n_test):
            n_node = random.randint(3, 10)
            deviation_constrain = random.uniform(1, 3)
            h_para = self.utl.randomly_generate_hemodynamic_parameters(n_node, deviation_constrain)
            temp = self.utl.check_proper_hemodynamic_parameters(h_para, deviation_constrain)
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

    def test_randomly_generate_u(self):
        if_plot = True
        n_node = self.utl.sample_node_number()
        n_stimuli = self.utl.sample_stimuli_number(n_node)
        t_time_point = self.utl.sample_scan_time()
        t_delta = self.utl.sample_t_delta()
        para_temp = t_time_point/t_delta
        n_time_point = mth.ceil(para_temp/32) * 32

        u = self.utl.randomly_generate_u(n_stimuli, n_time_point, t_delta)
        if if_plot:
            x = np.arange(n_time_point)
            for n in range(n_stimuli):
                y = u[:, n]
                plt.subplot(n_stimuli, 1, n+1)
                plt.plot(x, y)



class ParameterGraph_tests(unittest.TestCase):
    def setUp(self):
        self.pg = toolboxes.ParameterGraph()
        self.pgt = toolboxes.ParameterGraph()
        self.pgt._para_forerunner = {"if_random_l0": [],
                                    "l1a": ["if_random_l0"],
                                    "l1b": ["if_random_l0"],
                                    "l2a": ["l1a"],
                                    "l2b": ["l1a", "l1b"]}
        self.pgt.para_descendant = {"if_random_l0": ["l1a", "l1b"],
                                    "l1a": ["l2a", "l2b"],
                                    "l1b": ["l2b"],
                                    "l2a": [],
                                    "l2b": []}
        level_para = {
            "level_0": ["if_random_l0"],
            "level_1": ["l1a", "l1b"],
            "level_2": ["l2a", "l2b"]
        }
        level_para_order = ['level_0', 'level_1', 'level_2']
        self.pgt._level_para = OrderedDict(level_para, level_para_order)
        self.pgt.check_parameter_relation()
        self.pgt.para_level = {"if_random_l0": "level_0",
                               "l1a": "level_1",
                               "l1b": "level_1",
                               "l2a": "level_2",
                               "l2b": "level_2"}
        self.pgt.para_level_index = {"if_random_l0": 0,
                               "l1a": 1,
                               "l1b": 1,
                               "l2a": 2,
                               "l2b": 2}

        category_para = {1: ["if_random_l0"],
                         2: ["l1a", "l1b"],
                         3: ["l2a", "l2b"]}
        category_order = [1, 2, 3]
        self.pgt.category_para = OrderedDict(category_para, category_order)

        self.pgt.para_category = {"if_random_l0": 1,
                                  "l1a": 2,
                                  "l1b": 2,
                                  "l2a": 3,
                                  "l2b": 3}

    def tearDown(self):
        del self.pg
        del self.pgt

    def test_check_parameter_relation(self):
        self.assertTrue(self.pgt.check_parameter_relation())
        self.assertTrue(self.pg.check_parameter_relation())

    def test_get_para_forerunner_mapping(self):
        para_forerunner = self.pgt.get_para_forerunner_mapping()
        self.assertEqual(para_forerunner, self.pgt._para_forerunner)
        para_forerunner = self.pg.get_para_forerunner_mapping()

    def test_get_para_descendant_mapping(self):
        para_descendant = self.pgt.get_para_descendant_mapping()
        for key in para_descendant:
            self.assertEqual(set(para_descendant[key]), set(self.pgt.para_descendant[key]))
        para_descendant = self.pg.get_para_descendant_mapping()

    def test_get_level_para_mapping(self):
        level_para = self.pgt.get_level_para_mapping()
        self.assertEqual(level_para, self.pgt._level_para)

        level_para = self.pg.get_level_para_mapping()

    def test_get_para_level_mapping(self):
        if_print = False
        para_level = self.pgt.get_para_level_mapping()
        self.assertEqual(para_level, self.pgt.para_level)
        para_level = self.pg.get_para_level_mapping()
        if if_print:
            print(para_level)

    def test_get_para_level_index_mapping(self):
        para_level = self.pgt.get_para_level_index_mapping()
        self.assertEqual(para_level, self.pgt.para_level_index)
        para_level = self.pg.get_para_level_index_mapping()



    def test_get_para_category_mapping(self):
        if_print = False
        para_category = self.pgt.get_para_category_mapping()
        self.assertEqual(para_category, self.pgt.para_category)
        para_category = self.pg.get_para_category_mapping()

    def test_get_category_para_mapping(self):
        if_print = False
        category_para = self.pgt.get_category_para_mapping()
        for key in category_para:
            self.assertEqual(set(category_para[key]), set(self.pgt.category_para[key]))
        category_para = self.pg.get_category_para_mapping()
        if if_print:
            print(category_para)

    def test_abstract_flag(self):
        self.assertEqual(self.pg.abstract_flag(['if_random_stimuli_number', 'n_node', 'initializer']),
                         'if_random_stimuli_number',
                         'fail to find the correct flag')
        self.assertEqual(self.pg.abstract_flag([]),
                         None,
                         'fail to handle empty list')
        self.assertEqual(self.pg.abstract_flag(['n_node', 'initializer']),
                         None,
                         'fail to handle empty list')
        with self.assertRaises(ValueError):
            self.assertEqual(self.pg.abstract_flag(['if_random_stimuli_number', 'n_node',
                                                    'initializer', 'if_random_delta_t']),
                             'no_flag',
                             'fail to multiple flags')

    def test_get_flag(self):
        self.assertEqual(self.pg.get_flag_name('n_node'), 'if_random_node_number')
        self.assertEqual(self.pg.get_flag_name('hemodynamic_parameter'), 'if_random_hemodynamic_parameter')

    def test_make_graph(self):
        self.pgt.make_graph(self.pgt.get_para_descendant_mapping(),
                            file_name="../tests/test_output/ParameterGraph_make_graph_test",
                            rank_dict=self.pgt.get_level_para_mapping(),
                            rank_order=sorted(self.pgt.get_level_para_mapping().keys())
                            )
        if_update_graph = True
        if if_update_graph:
            self.pg.make_graph()

    def test_get_para_names(self):
        para_names = self.pgt.get_all_para_names()
        self.assertEqual(set(para_names), set(self.pgt._para_forerunner.keys()))
        para_names = self.pg.get_all_para_names()
        para_level_index = self.pg.get_para_level_index_mapping()
        for idx, value in enumerate(para_names[1:]):
            self.assertLessEqual(para_level_index[para_names[idx]],
                                 para_level_index[value])
        para_descendant = self.pg.get_para_descendant_mapping()
        self.assertEqual(set(para_names), set(para_descendant.keys()))

    def test_if_valid_para(self):
        self.assertTrue(self.pg.check_valid_para('A'))
        with self.assertRaises(ValueError):
            self.pg.check_valid_para('AA')


class Scanner_tests(unittest.TestCase):
    def setUp(self):
        self.sc = toolboxes.Scanner()
        '''
        self.du = toolboxes.DataUnit()
        self.du._secured_data['if_random_node_number'] = True
        self.du._secured_data['if_random_delta_t'] = True
        self.du._secured_data['if_random_scan_time'] = True
        self.du._secured_data['learning_rate'] = 0.1
        self.du._secured_data['n_backpro'] = 12
        self.du.complete_data_unit()
        '''

    def tearDown(self):
        del self.sc
        # del self.du

    def test_scan_x(self):
        '''
        parameter_package = self.du.collect_parameter_for_x_scan()
        x = self.sc.scan_x(parameter_package)
        self.assertTrue(self.sc.if_proper_x(x))
        '''
        # test case 1
        n_node = 3
        n_stimuli = 1
        n_time_point = 2
        Wxx = np.eye(n_node)
        Wxxu = [np.eye(n_node) for _ in range(n_stimuli)]
        Wxu = np.eye(n_node, n_stimuli)
        initial_x_state = np.zeros(n_node)
        u = np.zeros((n_time_point, n_stimuli))
        u[0, :] = 1
        x_correct = np.zeros((n_time_point, n_node))
        x_correct[1, 0] = 1
        parameter_package = {'Wxx': Wxx,
                             'Wxxu': Wxxu,
                             'Wxu': Wxu,
                             'initial_x_state': initial_x_state,
                             'u': u}
        x = self.sc.scan_x(parameter_package)
        np.testing.assert_array_equal(x, x_correct)

        # test Wxx
        n_node = 3
        n_stimuli = 1
        n_time_point = 2
        Wxx = np.eye(n_node)
        Wxxu = [np.eye(n_node) for _ in range(n_stimuli)]
        Wxu = np.eye(n_node, n_stimuli)
        initial_x_state = np.ones(n_node)
        u = np.zeros((n_time_point, n_stimuli))
        x_correct = np.ones((n_time_point, n_node))
        parameter_package = {'Wxx': Wxx,
                             'Wxxu': Wxxu,
                             'Wxu': Wxu,
                             'initial_x_state': initial_x_state,
                             'u': u}
        x = self.sc.scan_x(parameter_package)
        np.testing.assert_array_equal(x, x_correct)



    def test_scan_h(self):
        # parameter_package = self.du.collect_parameter_for_h_scan()
        # self.sc.scan_h(parameter_package)
        pass


class DataUnit_tests(unittest.TestCase):
    def setUp(self):
        self.du = toolboxes.DataUnit()
        self.dut = toolboxes.DataUnit()

    def tearDown(self):
        del self.du
        del self.dut

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
        with self.assertRaises(ValueError):
            self.du.set('Wxx', 400)
        self.du._secured_data['Wxx'] = 100
        self.du._secured_data['if_random_neural_parameter'] = False
        with self.assertRaises(ValueError):
            self.du.set('A', 400)

    def test_if_has_value(self):
        pass


if __name__ == '__main__':
    unittest.main()
