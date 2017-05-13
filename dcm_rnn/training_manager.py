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


    def prepare_data(self, du, dr):
        """
        Prepare data in du for dr in training. Create a 'data' dictionary 
        :param du: a DataUnit instance, with needed data
        :param dr: a DcmRnn instance, with needed parameters
        :return: 
        """
        self.data = {}

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
                warnings.warn("self.N_SEGMENTS is larger than the length of available self.data", UserWarning)
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

