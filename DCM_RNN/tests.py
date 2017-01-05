import unittest
import numpy as np
from DCM_RNN import utilities
import random


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.utl = utilities.Utilities()
        self.if_print = False

    def tearDown(self):
        del self.utl

    def test_randomly_generate_A_matrix(self):
        for n in range(10):
            size = random.randint(3, 10)
            A = self.utl.randomly_generate_A_matrix(size)
            w, v = np.linalg.eig(A)
            if self.if_print:
                print('created A matrix:')
                print(A)
                print('max real part = ' + str(max(w.real)))
            self.assertTrue(max(w)<0,'eigenvalues of A are not all negtive')


if __name__ == '__main__':
    unittest.main()
