# tests on data generation which tend to be time consuming
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


class DataUnit_tests(unittest.TestCase):
    def setUp(self):
        self.du = toolboxes.DataUnit()
        self.dut = toolboxes.DataUnit()

    def tearDown(self):
        del self.du
        del self.dut

    def test_complete_data_unit(self):
        self.du._secured_data['if_random_node_number'] = True
        self.du._secured_data['if_random_delta_t'] = True
        self.du._secured_data['if_random_scan_time'] = True
        self.du._secured_data['learning_rate'] = 0.1
        self.du._secured_data['n_backpro'] = 12
        # self.du.complete_data_unit()
        # print(self.du._secured_data)

if __name__ == '__main__':
    unittest.main()
