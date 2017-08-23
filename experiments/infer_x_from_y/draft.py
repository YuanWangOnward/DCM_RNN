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


class TrainingManager(tb.Initialization):

    def prepare_data(self, du, dr, data_package):
        """
        Prepare spm_data in du for dr in training. Create a 'spm_data' dictionary
        :param du: a DataUnit instance, with needed spm_data
        :param dr: a DcmRnn instance, with needed parameters
        :param data_package: a dictionary, stores configure and spm_data for a particular experimental case
        :return: modified
        """
        dp = data_package
        data = dp.data

        # create spm_data according to flag
        if dp.IF_RANDOM_H_PARA:
            dp.H_PARA_INITIAL = \
                dp.randomly_generate_hemodynamic_parameters(dr.n_region, deviation_constraint=2).astype(np.float32)
        else:
            dp.H_PARA_INITIAL = dp.get_standard_hemodynamic_parameters(n_node=dr.n_region).astype(np.float32)

        if dp.IF_RANDOM_H_STATE_INIT:
            dp.H_STATE_INITIAL = du.get('h')[random.randint(64, du.get('h').shape[0] - 64)].astype(np.float32)
        else:
            dp.H_STATE_INITIAL = \
                dp.set_initial_hemodynamic_state_as_inactivated(n_node=dr.n_region).astype(np.float32)

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
                warnings.warn("dp.MAX_SEGMENTS is larger than the length of available spm_data", UserWarning)
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

        # saved dp.SEQUENCE_LENGTH = dr.n_recurrent_step + (len(spm_data['x_true']) - 1) * dr.shift_data
        # collect merged spm_data (without split and merge, it can be tricky to cut proper part from du)
        data['u_merged'] = tb.merge(data['u'], dr.n_recurrent_step, dr.shift_data)
        data['x_true_merged'] = tb.merge(data['x_true'], dr.n_recurrent_step, dr.shift_data)
        # x_hat is with extra wrapper for easy modification with a single index
        data['x_hat_merged'] = \
            tb.ArrayWrapper(np.zeros(data['x_true_merged'].shape), dr.n_recurrent_step, dr.shift_data)
        data['h_true_monitor_merged'] = \
            tb.merge(data['h_true_monitor'], dr.n_recurrent_step, dr.shift_data)
        data['y_true_merged'] = tb.merge(data['y_true'], dr.n_recurrent_step, dr.shift_data)
        data['y_train_merged'] = tb.merge(data['y_train'], dr.n_recurrent_step, dr.shift_data)

        # run forward pass with x_true to show y error caused by error in the network parameters
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y_hat_x_true_log, h_hat_x_true_monitor_log, h_hat_x_true_connector_log = \
                dr.run_initializer_graph(sess, dp.H_STATE_INITIAL, data['x_true'])
        data['h_hat_x_true_monitor'] = h_hat_x_true_monitor_log
        data['y_hat_x_true'] = y_hat_x_true_log
        data['h_hat_x_true_monitor_merged'] = \
            tb.merge(h_hat_x_true_monitor_log, dr.n_recurrent_step, dr.shift_data)
        data['y_hat_x_true_merged'] = tb.merge(y_hat_x_true_log, dr.n_recurrent_step, dr.shift_data)

        data['loss_x_normalizer'] = np.sum(data['x_true_merged'].flatten() ** 2)
        data['loss_y_normalizer'] = np.sum(data['y_true_merged'].flatten() ** 2)
        data['loss_smooth_normalizer'] = np.std(data['x_true_merged'].flatten()) ** 2

        return data_package