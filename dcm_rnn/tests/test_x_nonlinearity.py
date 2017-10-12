from unittest import TestCase
import tensorflow as tf
from dcm_rnn import toolboxes as tb
from dcm_rnn.tf_model import DcmRnn
import numpy as np
import dcm_rnn.tf_model as tfm
import tensorflow as tf
import os
import random
import sys
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    sys.path.append('dcm_rnn')
elif '/share/apps/python3/' in sys.executable:
    PROJECT_DIR = '/home/yw1225/projects/DCM_RNN'
    print("It seems a remote run on NYU HPC")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib
    matplotlib.use('agg')
else:
    PROJECT_DIR = '.'
    print("Not sure executing machine. Make sure to set PROJECT_DIR properly.")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib
import scipy.io as sio

EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'estimation_x_nonlinearity_simulated_data')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'dcm_rnn_initial.mat')
TEMPLATE_PATH = os.path.join(PROJECT_DIR, 'experiments', 'compare_estimation_with_simulated_data', 'data', 'du_DCM_RNN.pkl')


class TestDcmRnnMainGraph(TestCase):

    spm_data = sio.loadmat(RAW_DATA_PATH)
    spm_data['stimulus_names'] = ['Photic', 'Motion', 'Attention']
    spm_data['node_names'] = ['V1', 'V5', 'SPC']
    spm_data['u'] = spm_data['u'].todense()
    # du = tb.load_template(TEMPLATE_PATH)

    '''
    dr = DcmRnn()
    print(os.getcwd())
    data_path = '../resources/template0.pkl'
    du = tb.load_template(data_path)
    dr.collect_parameters(du)
    dr.n_recurrent_step = 4
    neural_parameter_initial = {'A': du.get('A'), 'B': du.get('B'), 'C': du.get('C')}
    dr.build_main_graph(neural_parameter_initial=neural_parameter_initial)
    isess = tf.InteractiveSession()
    '''


    def setUp(self):
        # If it was not setup yet, do it
        '''
        if not self.CLASS_IS_SETUP:
            print('Setting up testing environment.')
            # run the real setup
            self.setupClass()
            # remember that it was setup already
            self.__class__.CLASS_IS_SETUP = True
        '''
        # self.isess.run(tf.global_variables_initializer())
        pass

    def tearDown(self):
        pass

    def test_simulation(self):
        data_path = '../resources/template0.pkl'
        du = tb.load_template(data_path)
        print(du.get('A'))
        print(du.get('B'))
        print(du.get('C'))
        print(du.get('hemodynamic_parameter'))

        print(du._secured_data.keys())



