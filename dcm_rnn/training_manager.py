import random
import numpy as np
import scipy.ndimage
import pandas as pd
import math as mth
import scipy as sp
import scipy.stats
import os
import sys
import warnings
import collections
import matplotlib.pyplot as plt
import subprocess
import pickle
import copy
import toolboxes as tb
import tf_model as tfm
import tensorflow as tf
from operator import attrgetter


class DistributedDataPackage:
    pass


class TrainingManager(tb.Initialization):
    """
    This class is used to the manage training process
    """
    def __init__(self):
        tb.Initialization.__init__(self)

        # global setting
        self.IF_RANDOM_H_PARA = False
        self.IF_RANDOM_H_STATE_INIT = False
        self.IF_NOISED_Y = False

        self.IF_NODE_MODE = False
        self.IF_IMAGE_LOG = True
        self.IF_DATA_LOG = True

        self.SNR = 3
        self.NODE_INDEX = 0
        self.SMOOTH_WEIGHT = 0.2
        self.N_RECURRENT_STEP = 64
        self.MAX_EPOCHS = 4

        self.MAX_EPOCHS_INNER = 4
        self.N_SEGMENTS = 128  # total amount of self.data segments
        # self.CHECK_STEPS = 4
        self.CHECK_STEPS = self.N_SEGMENTS * self.MAX_EPOCHS_INNER
        self.LEARNING_RATE = 128 / self.N_RECURRENT_STEP
        self.DATA_SHIFT = 4
        self.LOG_EXTRA_PREFIX = ''

        self.data = {}

    def __dir__(self):
        return ['IF_RANDOM_H_PARA', 'IF_RANDOM_H_STATE_INIT', 'IF_NOISED_Y',
                'IF_NODE_MODE', 'IF_IMAGE_LOG', 'IF_DATA_LOG',
                'SNR', 'NODE_INDEX', 'SMOOTH_WEIGHT', 'N_RECURRENT_STEP', 'MAX_EPOCHS',
                'MAX_EPOCHS_INNER', 'N_SEGMENTS', 'CHECK_STEPS', 'LEARNING_RATE', 'DATA_SHIFT',
                'LOG_EXTRA_PREFIX', 'data'
                ]

    def prepare_dcm_rnn(self, dr, tag='initializer'):
        """
        Set parameters in dr with training manager configures. Modify dr in place.
        :param dr: a DcmRnn instance, which will be used to build tensorflow model
        :param tag: experiment type
        :return: modified dr
        """
        dr.learning_rate = self.LEARNING_RATE
        dr.shift_data = self.DATA_SHIFT
        dr.n_recurrent_step = self.N_RECURRENT_STEP

        if tag == 'initializer':
            dr.loss_weighting['smooth'] = self.SMOOTH_WEIGHT
            if self.IF_NODE_MODE:
                dr.n_region = 1
            for key in dr.trainable_flags.keys():
                # in the initialization graph, the hemodynamic parameters are not trainable
                dr.trainable_flags[key] = False
        return dr

    '''
    def prepare_data(self, du, dr):
        """
        Prepare data in du for dr in training. Create a 'data' dictionary 
        :param du: a DataUnit instance, with needed data
        :param dr: a DcmRnn instance, with needed parameters
        :return: 
        """

        # create data according to flag
        if self.IF_RANDOM_H_PARA:
            self.H_PARA_INITIAL = \
                self.randomly_generate_hemodynamic_parameters(dr.n_region, deviation_constraint=2).astype(np.float32)
        else:
            self.H_PARA_INITIAL = self.get_standard_hemodynamic_parameters(n_node=dr.n_region).astype(np.float32)

        if self.IF_RANDOM_H_STATE_INIT:
            self.H_STATE_INITIAL = du.get('h')[random.randint(64, du.get('h').shape[0] - 64)].astype(np.float32)
        else:
            self.H_STATE_INITIAL = \
                self.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)

        if self.IF_NOISED_Y:
            std = np.std(du.get('y').reshape([-1])) / self.SNR
            self.NOISE = np.random.normal(0, std, du.get('y').shape)
        else:
            self.NOISE = np.zeros(du.get('y').shape)

        self.data['y_train'] = \
            tb.split(du.get('y') + self.NOISE, dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)
        max_segments_natural = len(self.data['y_train'])
        self.data['max_segments_natural'] = max_segments_natural
        self.data['y_true'] = \
            tb.split(du.get('y'), dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)[:max_segments_natural]
        self.data['h_true_monitor'] \
            = tb.split(du.get('h'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]
        self.data['x_true'] = tb.split(du.get('x'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]
        self.data['u'] = tb.split(du.get('u'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]

        if self.N_SEGMENTS is not None:
            if self.N_SEGMENTS > max_segments_natural:
                self.N_SEGMENTS = max_segments_natural
                warnings.warn("self.N_SEGMENTS is larger than the length of available data", UserWarning)
            else:
                self.data['u'] = self.data['u'][:self.N_SEGMENTS]
                self.data['x_true'] = self.data['x_true'][:self.N_SEGMENTS]
                self.data['h_true_monitor'] = self.data['h_true_monitor'][:self.N_SEGMENTS]
                self.data['y_true'] = self.data['y_true'][:self.N_SEGMENTS]
                self.data['y_train'] = self.data['y_train'][:self.N_SEGMENTS]

        if self.IF_NODE_MODE is True:
            node_index = self.NODE_INDEX
            self.data['x_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in self.data['x_true']]
            self.data['h_true_monitor'] = [np.take(array, node_index, 1) for array in self.data['h_true_monitor']]
            self.data['y_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in self.data['y_true']]
            self.data['y_train'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in self.data['y_train']]
            self.H_STATE_INITIAL = self.H_STATE_INITIAL[node_index].reshape(1, 4)

        # saved self.SEQUENCE_LENGTH = dr.n_recurrent_step + (len(self.data['x_true']) - 1) * dr.shift_data
        # collect merged self.data (without split and merge, it can be tricky to cut proper part from du)
        self.data['u_merged'] = tb.merge(self.data['u'], dr.n_recurrent_step, dr.shift_data)
        self.data['x_true_merged'] = tb.merge(self.data['x_true'], dr.n_recurrent_step, dr.shift_data)
        # x_hat is with extra wrapper for easy modification with a single index
        self.data['x_hat_merged'] = \
            tb.ArrayWrapper(np.zeros(self.data['x_true_merged'].shape), dr.n_recurrent_step, dr.shift_data)
        self.data['h_true_monitor_merged'] = \
            tb.merge(self.data['h_true_monitor'], dr.n_recurrent_step, dr.shift_data)
        self.data['y_true_merged'] = tb.merge(self.data['y_true'], dr.n_recurrent_step, dr.shift_data)
        self.data['y_train_merged'] = tb.merge(self.data['y_train'], dr.n_recurrent_step, dr.shift_data)

        # run forward pass with x_true to show y error caused by error in the network parameters
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y_hat_x_true_log, h_hat_x_true_monitor_log, h_hat_x_true_connector_log = \
                dr.run_initializer_graph(sess, self.H_STATE_INITIAL, self.data['x_true'])
        self.data['h_hat_x_true_monitor'] = h_hat_x_true_monitor_log
        self.data['y_hat_x_true'] = y_hat_x_true_log
        self.data['h_hat_x_true_monitor_merged'] = \
            tb.merge(h_hat_x_true_monitor_log, dr.n_recurrent_step, dr.shift_data)
        self.data['y_hat_x_true_merged'] = tb.merge(y_hat_x_true_log, dr.n_recurrent_step, dr.shift_data)

        self.data['loss_x_normalizer'] = np.sum(self.data['x_true_merged'].flatten() ** 2)
        self.data['loss_y_normalizer'] = np.sum(self.data['y_true_merged'].flatten() ** 2)
        self.data['loss_smooth_normalizer'] = np.std(self.data['x_true_merged'].flatten()) ** 2

        return self.data
    '''

    def prepare_distributed_data_package(self):
        ddp = DistributedDataPackage()
        for name in dir(self):
            setattr(ddp, name, copy.deepcopy(getattr(self, name)))
        #return {name: copy.deepcopy(getattr(self, name)) for name in dir(self)}
        return ddp

    def prepare_data(self, du, dr, data_package):
        """
        Prepare data in du for dr in training. Create a 'data' dictionary 
        :param du: a DataUnit instance, with needed data
        :param dr: a DcmRnn instance, with needed parameters
        :param data_package: a dictionary, stores configure and data for a particular experimental case
        :return: modified
        """
        dp = data_package
        data = dp.data

        # create data according to flag
        if dp.IF_RANDOM_H_PARA:
            dp.H_PARA_INITIAL = \
                dr.randomly_generate_hemodynamic_parameters(dr.n_region, deviation_constraint=2).astype(np.float32)
        else:
            dp.H_PARA_INITIAL = dr.get_standard_hemodynamic_parameters(n_node=dr.n_region).astype(np.float32)

        if dp.IF_RANDOM_H_STATE_INIT:
            dp.H_STATE_INITIAL = du.get('h')[random.randint(64, du.get('h').shape[0] - 64)].astype(np.float32)
        else:
            dp.H_STATE_INITIAL = \
                dr.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)

        if dp.IF_NOISED_Y:
            std = np.std(du.get('y').reshape([-1])) / dp.SNR
            dp.NOISE = np.random.normal(0, std, du.get('y').shape)
        else:
            dp.NOISE = np.zeros(du.get('y').shape)

        data['y_train'] = tb.split(du.get('y') + dp.NOISE, dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)
        max_segments_natural = len(data['y_train'])
        data['max_segments_natural'] = max_segments_natural
        data['y_true'] = tb.split(du.get('y'), dr.n_recurrent_step, dr.shift_data, dr.shift_x_y)[:max_segments_natural]
        data['h_true_monitor'] = tb.split(du.get('h'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]
        data['x_true'] = tb.split(du.get('x'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]
        data['u'] = tb.split(du.get('u'), dr.n_recurrent_step, dr.shift_data)[:max_segments_natural]

        if dp.N_SEGMENTS is not None:
            if dp.N_SEGMENTS > max_segments_natural:
                dp.N_SEGMENTS = max_segments_natural
                warnings.warn("dp.N_SEGMENTS is larger than the length of available data", UserWarning)
            else:
                data['u'] = data['u'][:dp.N_SEGMENTS]
                data['x_true'] = data['x_true'][:dp.N_SEGMENTS]
                data['h_true_monitor'] = data['h_true_monitor'][:dp.N_SEGMENTS]
                data['y_true'] = data['y_true'][:dp.N_SEGMENTS]
                data['y_train'] = data['y_train'][:dp.N_SEGMENTS]

        if dp.IF_NODE_MODE is True:
            node_index = dp.NODE_INDEX
            data['x_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['x_true']]
            data['h_true_monitor'] = [np.take(array, node_index, 1) for array in data['h_true_monitor']]
            data['y_true'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['y_true']]
            data['y_train'] = [array[:, node_index].reshape(dr.n_recurrent_step, 1) for array in data['y_train']]
            dp.H_STATE_INITIAL = dp.H_STATE_INITIAL[node_index].reshape(1, 4)

        # saved dp.SEQUENCE_LENGTH = dr.n_recurrent_step + (len(data['x_true']) - 1) * dr.shift_data
        # collect merged data (without split and merge, it can be tricky to cut proper part from du)
        data['u_merged'] = tb.merge(data['u'], dr.n_recurrent_step, dr.shift_data)
        data['x_true_merged'] = tb.merge(data['x_true'], dr.n_recurrent_step, dr.shift_data)
        # x_hat is with extra wrapper for easy modification with a single index
        data['x_hat_merged'] = \
            tb.ArrayWrapper(np.zeros(data['x_true_merged'].shape), dr.n_recurrent_step, dr.shift_data)
        data['h_true_monitor_merged'] = \
            tb.merge(data['h_true_monitor'], dr.n_recurrent_step, dr.shift_data)
        data['y_true_merged'] = tb.merge(data['y_true'], dr.n_recurrent_step, dr.shift_data)
        data['y_train_merged'] = tb.merge(data['y_train'], dr.n_recurrent_step, dr.shift_data)

        return data_package


    def train(self, dr, para_package, label):
        """"""
        print('Training starts!')
        # make a train scope copy of data
        data = para_package.data
        x_hat_previous = data['x_hat_merged'].data.copy()
        data['loss_x'] = []
        data['loss_y'] = []
        data['loss_smooth'] = []
        data['loss_total'] = []


        print("Optimization Finished!")

    def f(self, x):
        print(x)


