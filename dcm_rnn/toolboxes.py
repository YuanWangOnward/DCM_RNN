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
import re
from collections import Iterable
import tensorflow as tf
from math import factorial

def cdr(relative_path, if_print=False):
    file_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_path + relative_path)
    if if_print:
        print('working directory is ' + os.getcwd())


def load_template(data_path):
    current_dir = os.getcwd()
    print('working directory is ' + current_dir)
    with open(data_path, 'rb') as f:
        template = pickle.load(f)
    return template


def load_data(data_path):
    with open(data_path, 'rb') as f:
        template = pickle.load(f)
    return template


def save_data(data_path, data):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


def mse(value_hat, value_true=None):
    if value_true is not None:
        mse = ((value_hat.flatten() - value_true.flatten()) ** 2).mean()
    else:
        mse = (value_hat.flatten() ** 2).mean()
    return mse


def rmse(value_hat, value_true):
    error = value_hat.flatten() - value_true.flatten()
    norm_true = np.linalg.norm(value_true.flatten())
    norm_error = np.linalg.norm(error)
    return norm_error / norm_true


def argsort2(array):
    return np.squeeze(np.dstack(np.unravel_index(np.argsort(array.ravel()), array.shape)))


def take_value(array, rc_locations):
    return [array[x[0], x[1]] for x in rc_locations]


def split(data, n_segment, n_step=None, shift=0, split_dimension=0):
    """
    Split a large array spm_data into list of segments with step size n_step, ignoring the beginning shift points
    :param data:
    :param n_segment:
    :param n_step:
    :param shift:
    :param split_dimension:
    :param n_step:
    :return:
    """
    n_step = n_step or n_segment
    length = data.shape[split_dimension]
    data_shifted = np.take(data, range(shift, length), split_dimension)
    length_after_shift = length - shift
    output = []
    for i in range(0, length_after_shift - n_segment + 1, n_step):
        assert i + n_segment <= length_after_shift
        output.append(np.take(data_shifted, range(i, i + n_segment), split_dimension))
    return output


def split_index(data_shape, n_segment, n_step=None, shift=0, split_dimension=0):
    """
    Return a fancy index list so that corresponding segment can be manipulated easily
    :param data_shape:
    :param n_segment:
    :param n_step:
    :param shift:
    :param plit_dimension:
    :return:
    """
    n_step = n_step or n_segment
    length = data_shape[split_dimension]

    output = []
    for i in range(shift, length - n_segment + 1, n_step):
        assert i + n_segment <= length
        fancy_index = [slice(None)] * len(data_shape)
        start_point = i
        end_point = i + n_segment
        fancy_index[split_dimension] = range(start_point, end_point)
        output.append(fancy_index)
    return output


def merge(data_list, n_segment, n_step=None, merge_dimension=0):
    """
    :param data_list:
    :param n_segment:
    :param n_step:
    :param split_dimension:
    :return:
    """
    n_step = n_step or n_segment
    shape_output = list(data_list[0].shape)
    shape_output[merge_dimension] = (len(data_list) - 1) * n_step + n_segment
    output = np.zeros(shape_output)
    average_template = np.zeros(output.shape)
    fancy_index = [slice(None)] * output.ndim
    for i in range(len(data_list)):
        start_point = i * n_step
        end_point = start_point + n_segment
        fancy_index[merge_dimension] = range(start_point, end_point)
        output[fancy_index] += data_list[i]
        average_template[fancy_index] += 1.
    output = output / average_template
    return output


def split_data_for_initializer_graph(x_data, y_data, n_segment, n_step, shift_x_y):
    x_splits = split(x_data, n_segment, n_step=n_step)
    y_splits = split(y_data, n_segment, n_step=n_step, shift=shift_x_y)
    n_segments = min(len(x_splits), len(y_splits))
    x_splits = x_splits[:n_segments]
    y_splits = y_splits[:n_segments]
    return [x_splits, y_splits]


def make_a_batch(u, x, h, y):
    """
    Given lists of segments of u, x, h, and y, assemble a training batch.
    :param u:
    :param x:
    :param h:
    :param y:
    :return:
    """
    batch = {}
    batch['u'] = np.stack(u, axis=0)
    batch['x_initial'] = np.stack([v[0, :] for v in x], axis=0)
    batch['h_initial'] = np.stack([v[0, :, :] for v in h], axis=0)
    batch['y'] = np.stack(y, axis=0)
    return batch


def make_batches(u, x, h, y, batch_size=128, if_shuffle=True, extra=None):
    """
    Given u, x, h, and y as lists of data segments, assemble training batches.
    :param u:
    :param x:
    :param h:
    :param y:
    :param batch_size:
    :param if_shuffle:
    :param extra: a list of strings of extra outputs, ['u_previous', 'index']
    :return:
    """

    batches = []
    temp_u = copy.deepcopy(u)
    temp_x = copy.deepcopy(x)
    temp_h = copy.deepcopy(h)
    temp_y = copy.deepcopy(y)
    if extra is not None:
        if 'u_previous' in extra:
            temp_u_previous = copy.deepcopy(u)
            temp_u_previous.insert(0, np.zeros(temp_u_previous[0].shape))
            temp_u_previous.pop(-1)
        elif 'index' in extra:
            index = []

    if if_shuffle:
        idx = list(range(len(temp_y)))
        random.shuffle(idx)
        temp_u = [temp_u[i] for i in idx]
        temp_x = [temp_x[i] for i in idx]
        temp_h = [temp_h[i] for i in idx]
        temp_y = [temp_y[i] for i in idx]
        if extra is not None:
            if 'u_previous' in extra:
                temp_u_previous = [temp_u_previous[i] for i in idx]
            elif 'index' in extra:
                warnings.warn('Shuffle will cause index chaos.')


    for i in range(0, len(temp_y), batch_size):
        batch = make_a_batch(temp_u[i: i + batch_size],
                             temp_x[i: i + batch_size],
                             temp_h[i: i + batch_size],
                             temp_y[i: i + batch_size])

        if extra is not None:
            if 'u_previous' in extra:
                batch['u_previous'] = np.stack(temp_u_previous[i: i + batch_size], axis=0)
            elif 'index' in extra:
                batch['index'] = np.array(range(i, i + batch_size))
        # get rid of batches of less sample than batch size
        warnings.warn('Batch that does not meet batch size will be removed.')
        if all([len(batch[k]) == batch_size for k in batch.keys()]):
            batches.append(batch)

    return batches


def random_drop(batches, ration=0.5):
    """
    Randomly drop a ration of batches to create some randomness of gradient.
    :param batches:
    :param ration:
    :return:
    """
    random.shuffle(batches)
    return batches[: int(len(batches) * ration)]


def solve_for_effective_connection(x, u, prior=None):
    """
    Solve for the effective connection matrices Wxx, Wxxu, and Wxu from neural activity and stimuli
    x_t+1 = [Wxx Wxxu Wxu] * [x_t, Khatri-Rao_product(u_t, x_t), u_t]^T
    Y = W*X
    W = YX^T(XX^T)^(-1)
    :param x: neural activity, np.array with shape [n_time_point, n_node]
    :param u: stimuli, np.array with shape [n_time_point, n_stimuli]
    :param prior: a dictionary with keys in [Wxxu, Wxu]
    :return: return [Wxx Wxxu Wxu] where Wxxu is a list of np.arrays, others are np.array
    """
    # parameters
    n_time_point = x.shape[0]
    n_node = x.shape[1]
    n_stimuli = u.shape[1]

    # form Y
    Y = np.transpose(x[1:])
    assert Y.shape == (n_node, n_time_point - 1)

    # form X and modify Y
    X = [x[:-1]]
    Y_prior = np.zeros(Y.shape)
    xu = np.asarray([np.kron(u[i], x[i]) for i in range(n_time_point)])
    if prior is not None:
        if 'Wxxu' in prior.keys():
            Y_prior += np.matmul(np.concatenate(prior['Wxxu'], axis=1), np.transpose(xu[:-1]))
        else:
            X.append(xu[:-1])
        if 'Wxu' in prior.keys():
            Y_prior += np.matmul(prior['Wxu'], np.transpose(u[:-1]))
        else:
            X.append(u[:-1])
    else:
        X.append(xu[:-1])
        X.append(u[:-1])

    Y = Y - Y_prior
    X = np.transpose(np.concatenate(X, axis=1))

    # solve linear equation
    temp1 = np.matmul(Y, np.transpose(X))
    # assert temp1.shape == (n_node, n_node * (n_stimuli + 1) + n_stimuli)
    temp = np.matmul(X, np.transpose(X))
    print("Matrix condition number: " + str(np.linalg.cond(temp)))
    if np.linalg.cond(temp) < 1 / sys.float_info.epsilon:
        temp2 = np.linalg.inv(temp)
    else:
        warnings.warn("Ill conditioned equations")
        temp2 = np.linalg.pinv(temp)
    # assert temp2.shape == (n_node * (n_stimuli + 1) + n_stimuli, n_node * (n_stimuli + 1) + n_stimuli)
    W = np.matmul(temp1, temp2)

    # collect results
    Wxx = W[:, :n_node]
    if prior is not None:
        if 'Wxxu' in prior.keys():
            Wxxu = copy.deepcopy(prior['Wxxu'])
        else:
            Wxxu = []
            for s in range(n_stimuli):
                Wxxu.append(W[:, (s + 1) * n_node:(s + 2) * n_node])
        if 'Wxu' in prior.keys():
            Wxu = copy.deepcopy(prior['Wxu'])
        else:
            Wxu = W[:, -n_stimuli:]
    else:
        Wxxu = []
        for s in range(n_stimuli):
            Wxxu.append(W[:, (s + 1) * n_node:(s + 2) * n_node])
        Wxu = W[:, -n_stimuli:]
    return [Wxx, Wxxu, Wxu]


def ista(A, Y, alpha_mask=1, support=None, prior=None):
    """
    An implementation of the Iterative Soft Thresholding Algorithm (ista).
    It solves min_X 1/2||AX-Y\\^2_2 + ||alpha_mask.X||_1
    X^(k+1) = st(x^k - (1/L)AT(AX^k - Y), alpha_mask/L
    The (smallest) Lipschitz constant of the gradient ∇f is L(f) = λ_max(A^T*A)
    :param A: the mixing matrix
    :param Y: the target signal
    :param alpha_mask: a scalar or a np.array, if a scalar, the sparsity penalty is the same for each element of X;
                       if a np.array, it should have the same shape of X, which means the sparsity penalty of each
                       element of X is different.
    :param support: a binary np.array of the same shape of X. If an element is 0, the element in X at the corresponding
                    location is fixed as zeros; otherwise, can be any value.
    :param prior: a binary np.array of the same shape of X. Sparsity is with respect to value in prior
    :return: X, a np.array of coefficients
    """

    def soft_thresh(x, l, prior):
        return np.sign(x - prior) * np.maximum(np.abs(x - prior) - l, 0.) + prior

    # confirm the shape of X
    n_row = A.shape[1]
    n_col = Y.shape[1]
    if isinstance(alpha_mask, np.ndarray):
        assert alpha_mask.shape == (n_row, n_col)
    if support is not None:
        assert support.shape == (n_row, n_col)
    else:
        support = np.ones((n_row, n_col))
    if prior is not None:
        assert prior.shape == (n_row, n_col)
    else:
        prior = np.zeros((n_row, n_col))

    # find the Lipschitz constant
    L = max(sp.linalg.eigh(np.matmul(np.transpose(A), A), eigvals_only=True))
    step_size = 1 / L
    MAX_ITERATION = 500

    X_k = prior
    loss_list = []
    loss = mse(np.matmul(A, X_k), Y) + np.sum((alpha_mask * X_k).flatten())
    loss_list.append(loss)

    for iter in range(MAX_ITERATION):

        temp = np.matmul(A, X_k) - Y
        gradient = np.matmul(np.transpose(A), temp)

        # gredient descent
        X_temp = X_k - support * step_size * gradient

        # soft thresholding
        X_kp1 = soft_thresh(X_temp, alpha_mask / L, prior)

        # confirm support
        X_kp1 = support * X_kp1

        # show error
        loss = mse(np.matmul(A, X_kp1), Y) * np.sum((alpha_mask * X_kp1).flatten())
        loss_list.append(loss)
        # print(loss)


        # check stop
        if rmse(X_k, X_kp1) < 0.0001:
            print('.')
            break
        else:
            # for next step
            # print('.', end='', flush=True)
            X_k = X_kp1

    if iter == MAX_ITERATION - 1:
        warnings.warn("Hit MAX_ITERATION")

    # print('', end='', flush=True)
    loss_list = [v for v in loss_list if v > 0]
    plt.plot(np.log(loss_list))
    plt.xlabel('Iteration index')
    plt.ylabel('Total loss (ln)')
    return X_kp1


def setup_module():
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
    return PROJECT_DIR


def sigmoid(x):
    """
    Element wise sigmoid function.
    :param x: numpy array
    :return:
    """
    return 1 / (1 + np.exp(-x))

def generalized_sigmoid(x, generalization_parameters):
    """
    Element wise generalized sigmoid function.
    return y = vertical_zoom * sigmoid( horizontal_zoom * x + horizontal_shift) + vertical_shift
    """
    horizontal_zoom = generalization_parameters['horizontal_zoom']
    horizontal_shift = generalization_parameters['horizontal_shift']
    vertical_zoom = generalization_parameters['vertical_zoom']
    vertical_shift = generalization_parameters['vertical_shift']

    return vertical_zoom * (sigmoid(horizontal_zoom * x + horizontal_shift)) + vertical_shift


def neuron_fire_kernel(x_current, x_delta):
    x_current = np.array(copy.deepcopy(x_current))
    x_delta = np.array(x_delta)

    output = np.zeros(x_delta.shape)
    sign_mask = np.sign(x_current) == np.sign(x_delta)
    output[~sign_mask] = x_delta[~sign_mask]

    mask = x_current < 0
    x_current[mask] = x_current[mask] * 100
    x_current[~mask] = x_current[~mask] * 10
    weight = sigmoid(x_current) * (1 - sigmoid(x_current)) * 4

    output[sign_mask] = weight[sign_mask] * x_delta[sign_mask]

    return output

'''
def neuron_fire_kernel(x):
    x_temp = np.array(copy.deepcopy(x))
    mask = x_temp < 0
    x_temp[mask] = x_temp[mask] * 100
    x_temp[~mask] = x_temp[~mask] * 10
    return sigmoid(x_temp) * (1 - sigmoid(x_temp)) * 4
'''



class OrderedDict(collections.OrderedDict):
    def __init__(self, dictionary=None, key_order=None):
        if dictionary == None and key_order == None:
            super().__init__()
        else:
            super().__init__()
            if isinstance(key_order, dict):
                if set(dictionary.keys()) != set(key_order.keys()):
                    raise ValueError('Dictionary and order do not match')
                else:
                    self.key_order = self.form_list_order(key_order)
                    sorted_key_order = sorted(key_order.items(), key=lambda item: item[1])
                    sorted_key_order = [item[0] for item in sorted_key_order]
                    for key in sorted_key_order:
                        self[key] = dictionary[key]
            elif isinstance(key_order, list):
                if set(dictionary.keys()) != set(key_order):
                    raise ValueError('Dictionary and order do not match')
                else:
                    self.key_order = key_order
                    for key in key_order:
                        self[key] = dictionary[key]
            else:
                raise ValueError('Key order should be r list or dictionary')

    def form_list_order(self, dict_order):
        """
        OrderDict accepts dictionary type order. Internally, it uses list order.
        :param dict_order: {key:order} dictionary
        :return: [keys] list
        """
        sorted_key_order = sorted(dict_order.items(), key=lambda item: item[1])
        sorted_key_order = [item[0] for item in sorted_key_order]
        return sorted_key_order

    def get_order_index(self, key):
        """
        Find index in the order for given key.
        :param key: key
        :return: index
        """
        if key not in self.key_order:
            raise ValueError(key + ' is not r valid element in key order')
        else:
            return self.key_order.index(key)


class ArrayWrapper(np.ndarray):
    """
    Allow easy and fast access to np.array segment with simple index
    """

    def __new__(cls, array, segment_length, n_step=None, shift=0, split_dimension=0):
        return np.ndarray.__new__(cls, array.shape)

    def __init__(self, array, segment_length, n_step=None, shift=0, split_dimension=0):
        self.segment_length = segment_length
        self.n_step = n_step or segment_length
        self.shift = shift
        self.split_dimension = split_dimension
        self[:] = array[:]
        self.indices = split_index(self.data.shape, self.segment_length, self.n_step, self.shift, self.split_dimension)

    def __dir__(self):
        return np.ndarray.__dir__(self) + ['segment_length', 'n_step', 'shift', 'split_dimension', 'indices',
                                           'set', 'get']

    def set(self, index, data):
        self[self.indices[index]] = data

    def get(self, index=None):
        if index is None:
            return self.data
        else:
            return self[self.indices[index]]
'''
class ArrayWrapper(np.ndarray):
    """
    Allow easy and fast access to np.array segment with simple index
    """

    def __new__(cls, array, segment_length, n_step=None, shift=0, split_dimension=0):
        np.ndarray.__new__(array.shape)

    def __init__(self, array, segment_length, n_step=None, shift=0, split_dimension=0):
        self.data = array
        self.segment_length = segment_length
        self.n_step = n_step or segment_length
        self.shift = shift
        self.split_dimension = split_dimension
        print(tuple(array.shape))
        print(type(tuple(array.shape)))
        # self.__new__(self, tuple(array.shape))
        # np.ndarray.__new__(tuple(array.shape))

        self.indices = split_index(self.data.shape, self.segment_length, self.n_step, self.shift, self.split_dimension)

    def set(self, index, data):
        self.data[self.indices[index]] = data

    def get(self, index=None):
        if index == None:
            return self.data
        else:
            return self.data[self.indices[index]]
'''

class Initialization:
    def __init__(self,
                 n_node_low=None, n_node_high=None,
                 stimuli_node_ratio=None,
                 t_delta_low=None, t_delta_high=None,
                 scan_time_low=None, scan_time_high=None,
                 x_init_low=None, x_init_high=None,
                 s_init_low=None, s_init_high=None,
                 f_init_low=None, f_init_high=None,
                 v_init_low=None, v_init_high=None,
                 q_init_low=None, q_init_high=None,
                 A_off_diagonal_low=None, A_off_diagonal_high=None,
                 A_diagonal_low=None, A_diagonal_high=None,
                 A_generation_max_trial_number=None,
                 sparse_level=None,
                 B_init_low=None, B_init_high=None,
                 B_non_zero_probability=None,
                 B_sign_probability=None,
                 C_init_low=None, C_init_high=None,
                 u_type=None,
                 u_t_low=None, u_t_high=None,
                 u_amplitude=None,
                 u_interval_t_low=None, u_interval_t_high=None,
                 u_skip_rate=None,
                 u_frequency_low=None, u_frequency_high=None,
                 u_power_law_beta_low=None,
                 u_power_law_beta_high=None,
                 deviation_constraint=None,
                 h_parameter_check_statistics=None,
                 n_time_point_unit_length=None,
                 x_nonlinearity_type=None,
                 x_nonlinearity_parameter=None
                 ):

        self.n_node_low = n_node_low or 3
        self.n_node_high = n_node_high or 10
        self.stimuli_node_ratio = stimuli_node_ratio or 1 / 3
        self.t_delta_low = t_delta_low or 0.05
        self.t_delta_high = t_delta_high or 0.5
        self.scan_time_low = scan_time_low or 5 * 60  # in second
        self.scan_time_high = scan_time_high or 10 * 60  # in second

        self.x_init_low = x_init_low or 0
        self.x_init_high = x_init_high or 0.4
        self.s_init_low = s_init_low or -0.3
        self.s_init_high = s_init_high or 0.3
        self.f_init_low = f_init_low or 0.8
        self.f_init_high = f_init_high or 1.4
        self.v_init_low = v_init_low or 0.8
        self.v_init_high = v_init_high or 1.4
        self.q_init_low = q_init_low or 0.6
        self.q_init_high = q_init_high or 1.

        self.A_off_diagonal_low = A_off_diagonal_low or -0.8
        self.A_off_diagonal_high = A_off_diagonal_high or 0.8
        self.A_diagonal_low = A_diagonal_low or -1.0
        self.A_diagonal_high = A_diagonal_high or 0
        self.A_generation_max_trial_number = A_generation_max_trial_number or 5000
        self.sparse_level = sparse_level or 0.5
        self.B_init_low = B_init_low or -0.5
        self.B_init_high = B_init_high or 0.5
        self.B_non_zero_probability = B_non_zero_probability or 0.5
        self.B_sign_probability = B_sign_probability or 0.5
        self.C_init_low = C_init_low or 0.5
        self.C_init_high = C_init_high or 1.

        self.u_type_supported = ['box_train', 'sine', 'power_law']
        self.u_type = u_type or 'box_train'
        self.u_amplitude = u_amplitude or 0.1
        self.u_t_low = u_t_low or 5  # in second
        self.u_t_high = u_t_high or 10  # in second
        self.u_interval_t_low = u_interval_t_low or 5  # in second
        self.u_interval_t_high = u_interval_t_high or 10  # in second
        self.u_skip_rate = u_skip_rate or 0.2
        self.u_frequency_low = u_frequency_low or 0.01    # in Hz
        self.u_frequency_high = u_frequency_high or 0.1  # in Hz
        self.u_power_law_beta_low = u_power_law_beta_low or np.exp(1 - 3 / 8)
        self.u_power_law_beta_high = u_power_law_beta_high or np.exp(1 + 3 / 8)

        self.h_parameter_check_statistics = h_parameter_check_statistics or 'deviation'
        self.deviation_constraint = deviation_constraint or 1
        self.hemo_parameter_keys = ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0',
                                    'x_h_coupling']
        self.hemo_parameter_mean = pd.Series([0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 4., 0.03, 25, 40.3, 1.],
                                             self.hemo_parameter_keys)
        self.hemo_parameter_variance = pd.Series([0.0015, 0.0024, 0.015, 0.002, 0.0568, 0., 0., 0., 0., 0., 0.],
                                                 self.hemo_parameter_keys)

        self.n_time_point_unit_length = n_time_point_unit_length or 32


        self.x_nonlinearity_type = x_nonlinearity_type or 'None'
        if x_nonlinearity_parameter is None:
            y = np.array([0.05, 0.95])
            x = -np.log(1 / y - 1)
            vertical_zoom = 1 / (y[1] - y[0])
            horizontal_zoom = (x[1] - x[0])
            horizontal_shift = -0.5 * horizontal_zoom
            vertical_shift = - vertical_zoom * sigmoid(horizontal_shift)

            self.x_nonlinearity_parameter = {'None': {},
                                             'relu': {},
                                             'sigmoid': {'horizontal_zoom': horizontal_zoom,
                                                         'horizontal_shift': horizontal_shift,
                                                         'vertical_zoom': vertical_zoom,
                                                         'vertical_shift': vertical_shift
                                                         }}


    def sample_node_number(self):
        """
        :return: n_node: number of nodes (brain areas)
        """
        return np.random.randint(self.n_node_low, self.n_node_high)

    def sample_stimuli_number(self, n_node, stimuli_node_ratio=None):
        """
        Return number of stimuli, n_stimuli is sampled from np.random.randint(1, 1 + int(n_node * stimuli_node_ratio))
        :param n_node: number of nodes (brain areas)
        :param stimuli_node_ratio: the max n_stimuli n_node ratio
        :return: number of stimuli
        """
        stimuli_node_ratio = stimuli_node_ratio or self.stimuli_node_ratio
        return np.random.randint(1, 1 + int(n_node * stimuli_node_ratio))

    def sample_t_delta(self):
        """
        :return: t_delta: time interval for approximate differential equations
        """
        return np.random.uniform(self.t_delta_low, self.t_delta_high)

    def sample_scan_time(self):
        """
        :return: total scan time in second
        """
        return np.random.uniform(self.scan_time_low, self.scan_time_high)

    def randomly_initialize_connection_matrices(self, n_node, n_stimuli, sparse_level=None):
        """
        Generate r set of matrices for neural level equation x'=Ax+\sigma(xBu)+Cu.
        For assumptions about each matrix, please refer to each individual generating function.
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :param sparse_level: sparse level of off_diagonal elements, [0, 1],
        actual non-zeros elements equal int(sparse_level * (n_node-1) * n_node)
        :return: r dictionary with elements 'A', 'B', and 'C'
        """
        sparse_level = sparse_level or self.sparse_level
        connection_matrices = {'A': self.randomly_generate_sparse_A_matrix(n_node, sparse_level),
                               'B': self.randomly_generate_B_matrix(n_node, n_stimuli),
                               'C': self.randomly_generate_C_matrix(n_stimuli, n_stimuli)}
        return connection_matrices

    def roll(self, probability):
        """
        sample r random number from [0,1), if it's smaller than or equal to given probability, return True,
        otherwise, False
        :param probability: r number from [0 ,1]
        :return: boolean
        """
        if random.random() >= probability:
            return True
        else:
            return False

    def check_transition_matrix(self, A, max_eigenvalue=0.):
        """
        Check whether all the eigenvalues of A have negative real parts
        to ensure r corresponding linear system is stable
        :param A: candidate transition matrix
        :return: True or False
        """
        w, v = np.linalg.eig(A)
        if max(w.real) < max_eigenvalue:
            return True
        else:
            return False

    def randomly_generate_A_matrix(self, n_node, sparse_level=None):
        """
        Generate the A matrix for neural level equation x'=Ax+\sigma(xBu)+Cu.
        Eigenvalues of A must to negative to ensure the system is stable.
        A is sparse on off-diagonal places
        Assumption:
        Diagonal elements are negative meaning self-feedback is negative.
        Strength range of diagonal elements is (-0.5, 0].
        Strength range of off-diagonal elements is [-0.45, 0.45)
        :param n_node: number of nodes (brain areas)
        :param sparse_level: sparse level of off_diagonal elements, [0, 1],
        actual non-zeros elements equal int(sparse_level * (n_node-1) * n_node)
        :return: r sparse A matrix
        """
        sparse_level = sparse_level or self.sparse_level

        def get_a_sparse_matrix(n_node, sparse_level):
            A = np.random.uniform(self.A_off_diagonal_low, self.A_off_diagonal_high, size=(n_node, n_node))
            A = (A * (1 - np.identity(n_node))).flatten()
            sorted_indexes = np.argsort(np.abs(A), None)  # ascending order
            kept_number = int(sparse_level * (n_node - 1) * n_node)
            A[sorted_indexes[list(range(0, len(A) - kept_number))]] = 0
            A = A.reshape([n_node, n_node])
            A = A * (1 - np.identity(n_node)) + \
                np.diag(np.random.uniform(self.A_diagonal_low, self.A_diagonal_high, size=n_node))
            return A

        if sparse_level > 1 or sparse_level < 0:
            raise ValueError('Imporper sparse_level, it should be [0,1]')
        A = get_a_sparse_matrix(n_node, sparse_level)
        count = 0
        while not self.check_transition_matrix(A):
            A = get_a_sparse_matrix(n_node, sparse_level)
            count += 1
            if count > self.A_generation_max_trial_number:
                raise ValueError('Can not generate qualified A matrix with max trail number!')
        return A

    def randomly_generate_B_matrix(self, n_node, n_stimuli, if_resting_state=False):
        """
        Generate the B matrices for neural level equation x'=Ax+\sigma(xBu)+Cu.
        The number of B matrices equals the number of stimuli
        Assumptions:
        Each B matrix has only up to one non-zeros element, which means each stimulus only modify at most one edge
        Strength is in (-0.5, -0.2] U [0.2, 0.5)
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :return: r list of B matrices
        """
        if n_node < n_stimuli:
            raise ValueError('Improper number of nodes and stimuli')
        if if_resting_state:
            return [np.zeros((n_node, n_node)) for _ in range(n_stimuli)]
        else:
            B = []
            indexes = random.sample(range(0, n_node ** 2), n_node ** 2)
            for matrix_index in range(n_stimuli):
                B_current = np.zeros((n_node, n_node))
                if self.roll(self.B_non_zero_probability):
                    index_temp = indexes.pop()
                    row_index = int(index_temp / n_node)
                    column_index = index_temp % n_node
                    sign = -1 if self.roll(self.B_sign_probability) else 1
                    B_current[row_index, column_index] = sign * random.uniform(self.B_init_low, self.B_init_high)
                B.append(B_current)
            return B

    def randomly_generate_C_matrix(self, n_node, n_stimuli, if_resting_state=False):
        """
        Generate the C matrix for neural level equation x'=Ax+\sigma(xBu)+Cu.
        Assumptions:
        Each stimulus functions on only one brain area.
        Strength is in [0.5, 1)
        :param n_node: number of nodes (brain areas)
        :param n_stimuli: number of stimuli
        :return: r C matrix
        """
        if n_node < n_stimuli:
            raise ValueError('Improper number of nodes and stimuli')
        if if_resting_state:
            C = np.zeros((n_node, n_stimuli))
            for i in range(n_stimuli):
                C[i, i] = 1
            return C
        else:
            node_indexes = random.sample(range(0, n_node), n_stimuli)
            C = np.zeros((n_node, n_stimuli))
            for culumn_index, row_index in enumerate(node_indexes):
                C[row_index, culumn_index] = np.random.uniform(self.C_init_low, self.C_init_high)
            return C

    def randomly_generate_u(self, n_stimuli, n_time_point, t_delta, u_type=None):
        """
        Randomly generate input stimuli,
        :param n_stimuli: the amount of stimuli
        :param n_time_point: the total sample point of stimuli
        :param t_delta: time interval between adjacent sample points
        :return: np.array, random input, size (n_time_point, n_stimuli)
        """
        if u_type is None:
            u_type = self.u_type
        if u_type == 'box_train':
            def flip(num):
                if num is 0:
                    return 1
                if num is 1:
                    return 0

            u_t_low = self.u_t_low
            u_t_high = self.u_t_high
            u_n_low = int(u_t_low / t_delta)
            u_n_high = int(u_t_high / t_delta)

            u_interval_t_low = self.u_interval_t_low
            u_interval_t_high = self.u_interval_t_high
            u_interval_n_low = int(u_interval_t_low / t_delta)
            u_interval_n_high = int(u_interval_t_high / t_delta)

            u = np.zeros((n_time_point, n_stimuli))
            for n_s in range(n_stimuli):
                if n_s > 0:
                    i_current = np.random.randint(int((u_interval_n_low + u_interval_n_high) / 2))
                else:
                    i_current = 0
                value = 0
                while i_current < n_time_point:
                    if i_current > 0:
                        if random.random() > self.u_skip_rate:
                            value = flip(value)
                    else:
                        value = flip(value)
                    if value == 1:
                        if u_n_low == u_n_high:
                            step = u_n_low
                        else:
                            step = np.random.randint(u_n_low, u_n_high)
                    else:
                        if u_interval_n_low == u_interval_n_high:
                            step = u_interval_n_low
                        else:
                            step = np.random.randint(u_interval_n_low, u_interval_n_high)
                    i_next = i_current + step
                    if i_next >= n_time_point:
                        i_next = n_time_point
                    u[i_current:i_next, n_s] = value
                    i_current = i_next
            return u * self.u_amplitude
        elif u_type == 'sine':
            u = np.zeros((n_time_point, n_stimuli))
            t = np.array(range(n_time_point)) * t_delta
            n = 3
            for n_s in range(n_stimuli):
                frequencies = [np.random.uniform(self.u_frequency_low, self.u_frequency_high) for _ in range(n)]
                u[:, n_s] = np.sum([np.sin(2 * np.pi * frequency * t) for frequency in frequencies], axis=0) / n / 10
            return u * self.u_amplitude
        elif u_type == 'auto_regressor':
            pole = 0.99
            order = 1
            e = np.random.randn(n_time_point, n_stimuli) * 0.005

            a = [factorial(order) / factorial(i) / factorial(order - i) * np.power(pole, i) * np.power(-1., i + 1) for i
                 in range(1, order + 1)]
            a = [a[-i] for i in range(1, len(a) + 1)]
            a = np.array(a)
            a = np.tile(a, [n_stimuli, 1]).transpose()
            u = copy.deepcopy(e)
            for i in range(len(a), len(e)):
                u[i, :] = np.sum(u[i - len(a): i, :] * a, axis=0) + e[i, :]
            return u * self.u_amplitude
        else:
            raise ValueError(u_type + ' is not a proper stimuli input. Supported stimulus types include ' +
                             ' '.join(self.u_type_supported))

    def get_impulse_u(self, n_stimuli, n_time_point):
        """
        Generate impulse stimuli.
        :param n_stimuli: the amount of stimuli
        :param n_time_point: the total sample point of stimuli
        :return: np.array, size (n_time_point, n_stimuli)
        """
        u = np.zeros((n_time_point, n_stimuli))
        u[0, :] = 1
        return u

    def get_hemodynamic_parameter_prior_distributions(self):
        """
        Get prior distribution for hemodynamic parameters (Gaussian)
        :return: r pandas.dataframe containing mean, variance, and standard deviation of each parameter
        """
        prior_distribution = pd.DataFrame()
        prior_distribution['mean'] = self.hemo_parameter_mean
        prior_distribution['variance'] = self.hemo_parameter_variance
        prior_distribution['std'] = np.sqrt(self.hemo_parameter_variance)
        return prior_distribution.transpose()

    def get_expanded_hemodynamic_parameter_prior_distributions(self, n_node):
        """
        Repeat hemodynamic parameter prior distributions for each node and structure the results into pandas.dataframe
        :param n_node: number of nodes (brain areas)
        :return: r dict of  pandas.dataframe, containing hemodynamic parameter distributions for all the nodes,
                 distribution parameters include mean and standard deviation.
        """
        distributions = self.get_hemodynamic_parameter_prior_distributions()
        hemodynamic_parameters_mean = pd.DataFrame()
        hemodynamic_parameters_std = pd.DataFrame()
        for idx in range(n_node):
            hemodynamic_parameters_mean['region_' + str(idx)] = distributions.loc['mean']
            hemodynamic_parameters_std['region_' + str(idx)] = distributions.loc['std']
        return {'mean': hemodynamic_parameters_mean.transpose(), 'std': hemodynamic_parameters_std.transpose()}

    def get_standard_hemodynamic_parameters(self, n_node):
        """
        Get standard hemodynamic parameters, namely, the means of prior distribution of hemodynamic parameters
        :param n_node: number of nodes (brain areas)
        :return: r pandas cores frame, containing hemodynamic parameters for all the nodes
        """
        return self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)['mean']

    def randomly_generate_hemodynamic_parameters(self, n_node, deviation_constraint=None):
        """
        Get random hemodynamic parameters, sampled from prior distribution.
        The sample range is constrained to mean +/- deviation_constraint * standard_deviation
        :param n_node: number of nodes (brain areas)
        :param deviation_constraint: float, used to constrain the sample range
        :return: r pandas cores frame, containing hemodynamic parameters for all the nodes,
                 and optionally, normalized standard deviation.
        """
        deviation_constraint = deviation_constraint or self.deviation_constraint

        def sample_hemodynamic_parameters(hemodynamic_parameters_mean, hemodynamic_parameters_std,
                                          deviation_constraint):
            # sample r subject from hemodynamic parameter distribution
            h_mean = hemodynamic_parameters_mean
            h_std = hemodynamic_parameters_std

            h_para = copy.deepcopy(h_mean)
            h_devi = copy.deepcopy(h_std)

            p_shape = h_mean.shape
            for r in range(p_shape[0]):
                for c in range(p_shape[1]):
                    if h_std.iloc[r, c] > 0:
                        h_para.iloc[r, c] = np.random.normal(loc=h_mean.iloc[r, c], scale=h_std.iloc[r, c])
                        normalized_deviation = (h_para.iloc[r, c] - h_mean.iloc[r, c]) / h_std.iloc[r, c]
                        while abs(normalized_deviation) > deviation_constraint:
                            h_para.iloc[r, c] = np.random.normal(loc=h_mean.iloc[r, c], scale=h_std.iloc[r, c])
                            normalized_deviation = (h_para.iloc[r, c] - h_mean.iloc[r, c]) / h_std.iloc[r, c]
                    else:
                        pass
            return h_para

        temp = self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)
        hemodynamic_parameters_mean = temp['mean']
        hemodynamic_parameters_variance = temp['std']
        h_para = sample_hemodynamic_parameters(hemodynamic_parameters_mean,
                                               hemodynamic_parameters_variance,
                                               deviation_constraint)
        return h_para

    def evaluate_hemodynamic_parameters(self, hemodynamic_parameters, h_parameter_check_statistics=None):
        """
        For each hemodynamic parameter, map how it derives from the means.
        :param hemodynamic_parameters: r pandas.dataframe, containing sampled hemodynamic parameters
        :param h_parameter_check_statistics: specify which kind of statistic to check,
               it takes value in {'deviation', 'pdf'}
        :return: r pandas.dataframe, containing checking statistics
        """
        h_parameter_check_statistics = h_parameter_check_statistics or self.h_parameter_check_statistics
        n_node = hemodynamic_parameters.shape[0]
        temp = self.get_expanded_hemodynamic_parameter_prior_distributions(n_node)
        h_mean = temp['mean']
        h_std = temp['std']

        h_devi = copy.deepcopy(h_mean)
        h_pdf = copy.deepcopy(h_mean)
        h_para = hemodynamic_parameters

        p_shape = h_mean.shape
        for r in range(p_shape[0]):
            for c in range(p_shape[1]):
                if h_std.iloc[r, c] > 0:
                    h_devi.iloc[r, c] = (h_para.iloc[r, c] - h_mean.iloc[r, c]) / h_std.iloc[r, c]
                    h_pdf.iloc[r, c] = sp.stats.norm(0, 1).pdf(h_devi.iloc[r, c])
                else:
                    if h_para.iloc[r, c] == h_mean.iloc[r, c]:
                        h_devi.iloc[r, c] = 0.
                        h_pdf.iloc[r, c] = 1.
                    else:
                        raise ValueError('Hemodynamic parameter sampling goes wrong!')

        if h_parameter_check_statistics == 'deviation':
            return h_devi
        elif h_parameter_check_statistics == 'pdf':
            return h_pdf
        else:
            raise ValueError('Improper h_parameter_check_statistics value!')

    def check_proper_hemodynamic_parameters(self, hemodynamic_parameters, deviation_constraint=None):
        """
        Check if given hemodynamic parameters are within deviation constraint.
        If yes, return True; otherwise, throw error
        :param hemodynamic_parameters: r pandas.dataframe to be checked
        :param deviation_constraint: target deviation constraint, in normalized deviation,
        say if deviation_constraint=1, parameter with +/- one normalized deviation is acceptable
        :return: True or Error
        """
        deviation_constraint = deviation_constraint or self.deviation_constraint
        h_stat = self.evaluate_hemodynamic_parameters(hemodynamic_parameters, h_parameter_check_statistics='deviation')
        max_deviation = abs(h_stat).values.max()
        if max_deviation <= deviation_constraint:
            return True
        else:
            raise ValueError('The given hemodynamic parameters are not qualified by given deviation_constraint')

    def set_initial_neural_state_as_zeros(self, n_node):
        """
        Set initial neural state as zeros.
        :param n_node: number of nodes (brain areas)
        :return: zeros
        """
        return np.zeros(n_node)

    def randomly_generate_initial_neural_state(self, n_node):
        """
        Sample initial neural state. The distribution is set by experience.
        :param n_node: number of nodes (brain areas)
        :return: sampled initial neural state
        """
        return np.random.uniform(self.x_init_low, self.x_init_high, size=n_node)

    def set_initial_hemodynamic_state_as_inactivated(self, n_node):
        """
        Set initial hemodynamic state as inactivated, namely, s=0, f=1, v=1, q=1.
        :param n_node: number of nodes (brain areas)
        :return: array of initial hemodynamic state
        """
        h_state_initial = np.ones((n_node, 4))
        h_state_initial[:, 0] = 0
        return h_state_initial

    def randomly_generate_initial_hemodynamic_state(self, n_node):
        """
        Sample initial hemodynamic state. The distribution is set by experience.
        :param n_node: number of nodes (brain areas)
        :return: array of initial hemodynamic state
        """
        h_state_initial = np.ones((n_node, 4))
        h_state_initial[:, 0] = np.random.uniform(low=self.s_init_low, high=self.s_init_high, size=n_node)
        h_state_initial[:, 1] = np.random.uniform(low=self.f_init_low, high=self.f_init_high, size=n_node)
        h_state_initial[:, 2] = np.random.uniform(low=self.v_init_low, high=self.v_init_high, size=n_node)
        h_state_initial[:, 3] = np.random.uniform(low=self.q_init_low, high=self.q_init_high, size=n_node)
        return h_state_initial

    def calculate_dcm_rnn_x_matrices(self, A, B, C, t_delta):
        """
        Calculate matrices used in dcm_rnn for neural activity evolving.
        In DCM, neural level equation x'=Ax+\sigma(xBu)+Cu.
        In dcm_rnn, neural level equation x_t+1 = Wxx * x_t + \sigma(x_t * Wxxu * u) + Wxu * u
        :param A:
        :param B:
        :param C:
        :param t_delta: time interval for approximate differential equations in second
        :return: {'Wxx': Wxx, 'Wxxu': Wxxu, 'Wxu': Wxu}
        """
        n_node = A.shape[0]
        Wxx = np.array(A) * t_delta + np.eye(n_node, n_node, 0)
        Wxxu = [np.array(b) * t_delta for b in B]
        Wxu = C * t_delta
        return {'Wxx': Wxx, 'Wxxu': Wxxu, 'Wxu': Wxu}

    def calculate_dcm_rnn_h_matrices(self, hemodynamic_parameter, t_delta):
        """
        Calculate matrices used in dcm_rnn for hemodynamic evolving.
        :param hemodynamic_parameter: pd.dataframe
        :param t_delta: time interval for approximate differential equations in second
        :return: {'Whh': [Whh], 'Whx': [Whx], 'bh': [bh], 'Wo': [Wo], 'bo': [bo]}
        """
        n_node = hemodynamic_parameter.shape[0]
        Whh_ = []
        Whx_ = []
        Wo_ = []
        bh_ = []
        bo_ = []

        for n in range(n_node):
            # ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0']
            E0 = hemodynamic_parameter.loc['region_' + str(n), 'E0']
            k = hemodynamic_parameter.loc['region_' + str(n), 'k']
            gamma = hemodynamic_parameter.loc['region_' + str(n), 'gamma']
            tao = hemodynamic_parameter.loc['region_' + str(n), 'tao']
            epsilon = hemodynamic_parameter.loc['region_' + str(n), 'epsilon']
            V0 = hemodynamic_parameter.loc['region_' + str(n), 'V0']
            TE = hemodynamic_parameter.loc['region_' + str(n), 'TE']
            r0 = hemodynamic_parameter.loc['region_' + str(n), 'r0']
            theta0 = hemodynamic_parameter.loc['region_' + str(n), 'theta0']
            x_h_coupling = hemodynamic_parameter.loc['region_' + str(n), 'x_h_coupling']

            Whh = np.zeros((4, 7))
            Whh[0, 0] = -(t_delta * k - 1)
            Whh[0, 1] = -t_delta * gamma
            Whh[1, 0] = t_delta
            Whh[1, 1] = 1
            Whh[2, 1] = t_delta / tao
            Whh[2, 2] = 1
            Whh[2, 4] = -t_delta / tao
            Whh[3, 3] = 1
            Whh[3, 5] = -t_delta / tao
            Whh[3, 6] = t_delta / tao
            Whh_.append(Whh)

            Whx_.append(np.array([t_delta * x_h_coupling, 0, 0, 0]).reshape(4, 1))

            Wo = np.array([-(1 - epsilon) * V0, -4.3 * theta0 * E0 * V0 * TE, -epsilon * r0 * E0 * V0 * TE])
            Wo_.append(Wo)

            bh = np.array(np.asarray([t_delta * gamma, 0, 0, 0]).reshape(4, 1))
            bh_.append(bh)

            bo = V0 * (4.3 * theta0 * E0 * TE + epsilon * r0 * E0 * TE + (1 - epsilon))
            bo_.append(bo)

        return {'Whh': Whh_, 'Whx': Whx_, 'bh': bh_, 'Wo': Wo_, 'bo': bo_}

    def calculate_n_time_point(self, t_scan, t_delta):
        """
        Calculate n_time_point, n_time_point is ceil to self.n_time_point
        :param t_scan: total scan time in second
        :param t_delta: time interval for approximate differential equations in second
        :return: n_time_point
        """
        unit_length = self.n_time_point_unit_length
        para_temp = t_scan / t_delta
        n_time_point = mth.ceil(para_temp / unit_length) * unit_length
        return n_time_point


class ParameterGraph:
    def __init__(self):
        self._para_forerunner = {
            # inherited
            'initializer': [],
            'parameter_graph': [],
            'scanner': [],

            'if_random_neural_parameter': [],
            'if_random_hemodynamic_parameter': [],
            'if_random_x_state_initial': [],
            'if_random_h_state_initial': [],
            'if_random_stimuli': [],
            'if_random_node_number': [],
            'if_random_stimuli_number': [],
            'if_random_delta_t': [],
            'if_random_scan_time': [],
            'if_x_nonlinearity': [],
            'if_resting_state': [],

            'u_type': ['if_resting_state', 'initializer'],
            'x_nonlinearity_type': ['if_x_nonlinearity', 'initializer'],
            'n_node': ['if_random_node_number', 'initializer'],
            't_delta': ['if_random_delta_t', 'initializer'],
            't_scan': ['if_random_scan_time', 'initializer'],


            'n_time_point': ['t_scan', 't_delta'],
            'n_stimuli': ['if_random_stimuli_number', 'n_node', 'initializer'],
            'x_nonlinearity_parameter': ['x_nonlinearity_type', 'initializer'],


            'u': ['if_random_stimuli',
                  'u_type',
                  'n_stimuli',
                  'n_time_point',
                  't_delta',
                  'initializer'],
            'A': ['if_random_neural_parameter',
                  'n_node',
                  'initializer'],
            'B': ['if_random_neural_parameter',
                  'if_resting_state',
                  'n_node',
                  'n_stimuli',
                  'initializer'],
            'C': ['if_random_neural_parameter',
                  'if_resting_state',
                  'n_node',
                  'n_stimuli',
                  'initializer'],

            # 'alpha': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'E0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'k': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'gamma': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'tao': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'epsilon': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'V0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'TE': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'r0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # 'theta0': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],
            # they are all put in

            'hemodynamic_parameter': ['n_node', 'if_random_hemodynamic_parameter', 'initializer'],

            'initial_x_state': ['n_node', 'if_random_x_state_initial', 'initializer'],
            'initial_h_state': ['n_node', 'if_random_h_state_initial', 'initializer'],

            'Wxx': ['A', 't_delta'],
            'Wxxu': ['B', 't_delta'],
            'Wxu': ['C', 't_delta'],  # 'C' matrix equivalence in dcm_rnn model

            'Whh': ['hemodynamic_parameter', 't_delta'],
            'Whx': ['hemodynamic_parameter', 't_delta'],
            'bh': ['hemodynamic_parameter', 't_delta'],
            'Wo': ['hemodynamic_parameter'],
            'bo': ['hemodynamic_parameter'],

            'x': ["Wxx", "Wxxu", "Wxu", 'initial_x_state', 'u', 'scanner'],
            'h': ['Whh', 'Whx', 'bh', 'hemodynamic_parameter', 'x', 'scanner'],
            'y': ['Wo', 'bo', 'h', 'scanner'],
        }

        level_para = {
            'inherited': ['initializer', 'parameter_graph', 'scanner'],
            'level_0': ['if_random_neural_parameter',
                        'if_random_hemodynamic_parameter',
                        'if_random_x_state_initial',
                        'if_random_h_state_initial',
                        'if_random_stimuli',
                        'if_random_node_number',
                        'if_random_stimuli_number',
                        'if_random_delta_t',
                        'if_random_scan_time',
                        'if_x_nonlinearity',
                        'if_resting_state'
                        ],
            'level_1': ['n_node', 't_delta', 't_scan', 'x_nonlinearity_type', 'u_type'],
            'level_2': ['n_time_point',
                        'n_stimuli',
                        'A',
                        'hemodynamic_parameter',
                        'initial_x_state',
                        'initial_h_state',
                        'x_nonlinearity_parameter'],
            'level_3': ['u',
                        'B', 'C',
                        'Wxx', 'Whx', 'Whh', 'bh', 'Wo', 'bo'],
            'level_4': ['Wxxu', 'Wxu'],
            'level_5': ['x'],
            'level_6': ['h'],
            'level_7': ['y']
        }
        level_para_order = ['inherited', 'level_0', 'level_1', 'level_2', 'level_3', 'level_4', 'level_5',
                            'level_6', 'level_7']
        self._level_para = OrderedDict(level_para, level_para_order)

        self.check_parameter_relation()

    def check_parameter_relation(self):
        """
        Check if _para_forerunner and para_level are in consistence with each other, since they are hand-coded
        which is prone to error
        :return: True or raiseError
        """

        variable_level_map = {}
        for key, value in self._level_para.items():
            for val in value:
                variable_level_map[val] = self._level_para.get_order_index(key)
        for key, value in self._para_forerunner.items():
            if not value:
                if variable_level_map[key] not in [0, 1]:
                    raise ValueError(key + ' parameter graph error')
            else:
                temp = [variable_level_map[val] for val in value]
                max_forerunner_level = max(temp)
                if variable_level_map[key] <= max_forerunner_level:
                    raise ValueError(key + ' parameter graph error')
        return True

    def get_para_forerunner_mapping(self):
        return copy.deepcopy(self._para_forerunner)

    def forerunner2descendant(self, forerunner_mapping=None, key_order=None, if_complete=True):
        """
        Transfer {parameter:[forerunners]} mapping  dictionary to
        {parameter:[descendants]} mapping dictionary
        :param forerunner_mapping: parameter:forerunners mapping dictionary
        :param key_order: if an {key:order} dictionary is given, return an OderderDict, otherwise, r dict
        :param if_complete: if True, treat keys in forerunner dictionary as valid parameters.
            Parameter without r descendant will also be added to resulting dictionary
        :return: parameter: {parameter:[descendants]} mapping dictionary
        """
        forerunner_mapping = forerunner_mapping or self._para_forerunner
        descendant_mapping = {}
        for key, value in forerunner_mapping.items():
            for val in value:
                if val in descendant_mapping.keys():
                    descendant_mapping[val].append(key)
                else:
                    descendant_mapping[val] = [key]
        if if_complete is True:
            all_name = []
            for key, value in forerunner_mapping.items():
                all_name = all_name + [key]
                all_name = all_name + value
            all_name = set(all_name)
            for name in all_name:
                if name not in descendant_mapping.keys():
                    descendant_mapping[name] = []
        if key_order != None:
            if set(key_order.keys()) != set(descendant_mapping.keys()):
                raise ValueError('Given order does not match reversed dictionary.')
            else:
                return OrderedDict(descendant_mapping, key_order)
        else:
            return descendant_mapping

    def get_para_descendant_mapping(self):
        para_forerunner = self.get_para_forerunner_mapping()
        para_descendant = self.forerunner2descendant(para_forerunner)
        return para_descendant

    def level_para2para_level(self, level_para=None):
        """
        Transfer {level:[parameters]} mapping dictionary to
        {parameter:level} mapping dictionary
        :param level_para: {level:[parameters]} mapping dictionary
        :return: {parameter:level} mapping dictionary, if _level_para is an OrderedDict, return an OrderedDict
        """
        level_para = level_para or self._level_para
        para_level_temp = self.forerunner2descendant(level_para, if_complete=False)
        if isinstance(level_para, OrderedDict):
            para_level = OrderedDict()
            key_order = []
            for level in level_para.key_order:
                for para, value in para_level_temp.items():
                    if value[0] == level:
                        para_level[para] = level
                        key_order.append(para)
            para_level.key_order = key_order
        else:
            para_level = {}
            for key, value in para_level_temp.items():
                if len(value) > 1:
                    raise ValueError('One parameters cannot belong to more than one level.')
                else:
                    para_level[key] = value[0]
        return para_level

    def get_level_para_mapping(self):
        return copy.deepcopy(self._level_para)

    def get_para_level_mapping(self, level_para=None):
        """
        Get {parameter:level} dictionary from {level:[parameters]} dictionary
        :param level_para:{level:[parameters]} dictionary
        :return: {parameter:level} dictionary
        """
        level_para = level_para or self._level_para
        return self.level_para2para_level(level_para)

    def get_para_level_index_mapping(self, level_para=None, order=None):
        """
        Get {parameter:level_index} dictionary from {level:[parameters]} dictionary
        :param level_para:{level:[parameters]} dictionary
        :param order: [levels] list recording level order
        :return: {parameter:level} dictionary
        """
        level_para = level_para or self._level_para
        order = order or self._level_para.key_order
        para_level = self.get_para_level_mapping(level_para)
        return {key: order.index(value) for key, value in para_level.items()}

    def get_para_category_mapping(self, para_forerunner=None):
        """
        Base on prerequisites (forerunners), put parameters in to 3 categories.
        Category one: no prerequisites, should be assigned value directly
        Category two: with prerequisite and has if_random_ flag in prerequisites, one needs to check flag before assign
        Category three: with prerequisite and has no flag, should not be assigned directly, its value should be derived
            by its prerequisites.
        :param para_forerunner: r dictionary recording _para_forerunner of each parameter
        :return: r {para:category} dictionary recording category of each parameter
        """
        para_category = {}
        para_forerunner = para_forerunner or self._para_forerunner
        for key, value in para_forerunner.items():
            if not value:
                para_category[key] = 1
            else:
                flag = self.abstract_flag(value)
                if flag != None:
                    para_category[key] = 2
                else:
                    para_category[key] = 3
        return para_category

    def para_category2category_para(self, para_category, category_order=None):
        """
        Transform {para:category} mapping to {category:[paras]} mapping
        :param para_category: {para:category} mapping
        :param category_order: order of the category, if not specified, sorted() will be used
        :return: {category:[paras]} mapping, OrderedDict
        """
        category_para_temp = {}
        for key, value in para_category.items():
            if value in category_para_temp.keys():
                category_para_temp[value].append(key)
            else:
                category_para_temp[value] = [key]
        if category_order is None:
            category_order = sorted(category_para_temp.keys())
        category_para = OrderedDict(category_para_temp, category_order)
        return category_para

    def get_category_para_mapping(self, para_forerunner=None):
        para_forerunner = para_forerunner or self._para_forerunner
        para_category = self.get_para_category_mapping(para_forerunner)
        category_para = self.para_category2category_para(para_category)
        return category_para

    def abstract_flag(self, prerequisites):
        """
        Find if_ flag from the prerequisites of r parameter
        :param prerequisites: r list of parameters
        :return: None, if prerequisites is empty, or there is no flag in prerequisites
                 r flag name if there is r flag in prerequisites
        """
        if not prerequisites:
            return None
        else:
            temp = [s for s in prerequisites if 'if_' in s]
            if len(temp) == 0:
                return None
            elif len(temp) == 1:
                return temp[0]
            else:
                return temp


    def get_flag_name(self, para):
        """
         Find if_ flag from the forerunners of r parameter
        :param para: target parameter
        :return: None, if forerunners is empty, or there is no flag in forerunners
                 r flag name if there is r flag in forerunners
        """
        self.check_valid_para(para)
        para_forerunner = self.get_para_forerunner_mapping()
        return self.abstract_flag(para_forerunner[para])

    def make_graph(self, relation_dict=None, file_name=None, rank_dict=None, rank_order=None):
        """
        Create .gv file and then use dot tool to create r diagram.
        :param relation_dict: {form:[tos]} structure, recording edges
        :param file_name: title, file name of .gv and .png file
        :param rank_dict: {rank_name:[members]} dict recording rank information
        :param rank_order: [rank_names] list recording rank order
        :return: True if it runs to the end
        """
        '''
            if relation_dict == None and file_name == None and rank_dict == None and rank_order == None:
            relation_dict = self.forerunner2descendant(self.get_para_forerunner_mapping())
            file_name = "documents/parameter_level_graph"
            rank_dict = self._level_para
            rank_order = self._level_para.key_order
        '''
        relation_dict = relation_dict or self.forerunner2descendant(self.get_para_forerunner_mapping())
        file_name = file_name or "documents/parameter_level_graph"
        rank_dict = rank_dict or self._level_para
        rank_order = rank_order or self._level_para.key_order

        with open(file_name + ".gv", 'w') as f:
            f.write("digraph G {\n")
            f.write("          splines=ortho;\n")
            f.write("          fontsize = 48;\n")
            f.write("          rankdir = \"LR\";\n")
            f.write("          node[fontsize=24];\n")
            f.write("          edge[penwidth=2];\n")

            if rank_order != None:
                f.write("          {\n")
                f.write("          node [shape=plaintext fontsize=36];\n")
                for key in rank_order:
                    if key != rank_order[-1]:
                        f.write("          " + str(key) + " -> \n")
                    else:
                        f.write("          " + str(key) + "\n")
                f.write("          }\n")

            if rank_dict != None:
                for key, value in rank_dict.items():
                    f.write("          {rank = same;\n")
                    if type(key) is not str:
                        key = str(key)
                    f.write("          " + key + ";\n")
                    for val in value:
                        f.write("          " + val + ";\n")
                    f.write("          }\n")

            # relation edges
            for key, value in relation_dict.items():
                randome_coloar = ''.join([random.choice('0123456789ABCE') for x in range(6)])
                if not value:
                    # value is empty

                    string = "          " + key + " ;\n"
                else:
                    # value is not empty
                    string = ""
                    for val in value:
                        string = string + "          " + key + " -> " + val + " [color=\"#" + randome_coloar + "\"];\n"
                f.write(string)

            # add title
            f.write('          labelloc = "t";\n')
            f.write('          label = "' + file_name + '";\n')
            f.write("}")
            # call dot tool to draw diagram
            # however, it doesn't work at this moment
            source_file = file_name + ".gv"
            target_file = file_name + ".png"
            subprocess.run(["dot", "-Tpng", source_file, "-o", target_file], check=True)
            string = ' '.join(["dot", "-Tpng", source_file, "-o", target_file])
            print('run2 the following command in the terminal to update graph:')
            print('cd ' + os.getcwd())
            print(string)

    def get_all_para_names(self):
        """
        Get all parameter node_names in r DataUnit.
        :return: r list of parameter node_names, sorted by parameter level
        """
        para_level = self.get_para_level_mapping()
        return sorted(para_level.keys(), key=lambda key: para_level[key])

    def check_valid_para(self, para):
        """
        Check if r given parameter name is r valid one
        :param para: target name
        :return: True or raise error
        """
        valid_names = self.get_all_para_names()
        if para in valid_names:
            return True
        else:
            # print('current parameter name is ' + para)
            # string = ', '.join(valid_names)
            # print('all_para_names: ' + string)
            raise ValueError('Improper name.')


class Scanner:
    def __init__(self, snr_y=None,
                 x_max_bound=None, x_mean_bound=None,
                 x_var_low=None, x_var_high=None,
                 h_value_low=None, h_value_high=None):
        self.snr_y = snr_y or 2
        '''
        self.x_max_bound = x_max_bound or 8
        self.x_mean_bound = x_mean_bound or 2
        self.x_var_low = x_var_low or 0.1
        self.x_var_high = x_var_high or 4
        self.h_value_low = h_value_low or 0.125
        self.h_value_high = h_value_high or 8
        '''
        self.x_max_bound = x_max_bound or 2.
        self.x_mean_bound = x_mean_bound or 0.2
        self.x_var_low = x_var_low or 0.001
        self.x_var_high = x_var_high or 0.05
        self.h_value_low = h_value_low or 0.125
        self.h_value_high = h_value_high or 8


    def scan_x(self, parameter_package):
        """
        Calculate x, namely neural activity
        :param parameter_package: r dictionary containing all needed parameters
        :return: x, np.array, (n_time_point, n_node)
        """
        Wxx = parameter_package['Wxx']
        Wxxu = parameter_package['Wxxu']
        Wxu = parameter_package['Wxu']
        initial_x_state = parameter_package['initial_x_state']
        u = parameter_package['u']
        x_nonlinearity_type = parameter_package['x_nonlinearity_type']
        # x_nonlinearity_parameter = parameter_package['x_nonlinearity_parameter']

        n_node = Wxu.shape[0]
        n_stimuli = Wxu.shape[1]
        n_time_point = u.shape[0]
        x = np.zeros((n_time_point, n_node))
        x[0, :] = initial_x_state

        for i in range(1, n_time_point):
            if x_nonlinearity_type == 'sigmoid':
                tmp1 = np.matmul(Wxx - np.eye(n_node), x[i - 1, :])
            else:
                tmp1 = np.matmul(Wxx, x[i - 1, :])
            tmp2 = [np.matmul(Wxxu[idx], x[i - 1, :] * u[i - 1, idx]) for idx in range(n_stimuli)]
            tmp2 = np.sum(np.asarray(tmp2), 0)
            tmp3 = np.matmul(Wxu, u[i - 1, :])
            if x_nonlinearity_type == 'sigmoid':
                x[i, :] = x[i - 1, :] + neuron_fire_kernel(x[i - 1, :], tmp1 + tmp2 + tmp3)
            else:
                x[i, :] = tmp1 + tmp2 + tmp3
            if x_nonlinearity_type == 'None':
                pass
            else:
                if x_nonlinearity_type == 'relu':
                    x = np.maximum(x, 0)
                elif x_nonlinearity_type == 'sigmoid':
                    pass
        return x

    def if_proper_x(self, x):
        """
        Check if r x seems r good one in terms of some statistics which in includes:
        max absolute value and energy distribution on frequency
        :param x: neural activities, np.narray of (n_time_point, n_node)
        :return: True if x means proper; False otherwise
        """
        max_absolute_value = max(abs(x.flatten()))
        max_mean_value = np.max(abs(np.mean(x, axis=0)))
        n_time_point = x.shape[0]
        variance = np.var(x[int(n_time_point / 2):, :], 0)
        min_var = min(variance)
        max_var = max(variance)

        '''
        fx = np.fft.fft(x, x_axis=0) / np.sqrt(x.shape[0])
        low_frequency_range = int(x.shape[0]/20)
        low_frequency_energy = 1
        energy_persentage = 1
        '''

        if max_absolute_value > self.x_max_bound:
            # print('x max_abslute_value = ' + str(max_abslute_value))
            print("Not proper x: max value")
            return False
        if max_mean_value > self.x_mean_bound:
            print("Not proper x: mean value")
            return False
        if min_var < self.x_var_low or max_var > self.x_var_high:
            # print('x min_var = ' + str(min_var))
            # print('x max_var = ' + str(max_var))
            print("Not proper x: variance")
            return False
        return True

    def if_proper_h(self, h):
        """
        Check if r set of generated h seems r reasonable one.
        Namely, f, v, q should be within [self.h_value_low, self.h_value_high] all the time
        :param h: test hemodynamic state, np.array of (n_time_point, n_node, 4)
        :return: True if h means proper; False otherwise
        """
        h_temp = h[:, :, 1:].flatten()
        min_value = np.min(h_temp)
        max_value = np.max(h_temp)
        if min_value >= self.h_value_low and max_value <= self.h_value_high:
            return True
        else:
            print("Not proper h")
            return False

    def phi_h(self, h_state_current, alpha, E0):
        """
        used to map hemodynamic states into higher dimension
        for hemodynamic states scan
        :param h_state_current:
        :param alpha:
        :param E0:
        :return:
        """
        h_state_augmented = np.zeros(7)
        h_state_augmented[0:4] = h_state_current
        h_state_augmented[4] = h_state_current[2] ** (1 / alpha)
        h_state_augmented[5] = h_state_current[3] / (h_state_current[2]) * h_state_augmented[4]
        h_state_augmented[6] = (1 - (1 - E0) ** (1 / h_state_current[1])) / (E0) * h_state_current[1]
        return h_state_augmented

    def phi_h_parallel(self, h_state_current, alpha, E0):
        """
        used to map hemodynamic states into higher dimension
        for hemodynamic states scan
        :param h_state_current:
        :param alpha:
        :param E0:
        :return:
        """
        h_state_augmented = np.zeros((h_state_current.shape[0], 7))
        h_state_augmented[:, 0:4] = h_state_current
        h_state_augmented[:, 4] = h_state_current[:, 2] ** (1 / alpha)
        h_state_augmented[:, 5] = h_state_current[:, 3] / (h_state_current[:, 2]) * h_state_augmented[:, 4]
        h_state_augmented[:, 6] = (1 - (1 - E0) ** (1 / h_state_current[:, 1])) / (E0) * h_state_current[:, 1]
        return h_state_augmented

    def scan_h(self, parameter_package):
        """
        Calculate h, namely hemodynamic response
        :param parameter_package: r dictionary containing all needed parameters
        :return: np.array of size (n_time_point, n_node, 4)
        """
        hemodynamic_parameters = parameter_package['hemodynamic_parameter']
        Whh = parameter_package['Whh']
        bh = parameter_package['bh']
        Whx = parameter_package['Whx']
        initial_h_state = parameter_package['initial_h_state']
        x = parameter_package['x']

        n_time_point = x.shape[0]
        n_node = x.shape[1]
        eps = np.finfo(float).eps
        h = np.ones((n_time_point, n_node, 4))
        h[:, :, 0] = 0
        h[0, :, :] = initial_h_state

        for t in range(1, n_time_point):
            for n in range(0, n_node):
                alpha = hemodynamic_parameters.loc['region_' + str(n), 'alpha']
                E0 = hemodynamic_parameters.loc['region_' + str(n), 'E0']
                h_temp = (np.matmul(Whh[n], self.phi_h(h[t - 1, n, :], alpha, E0)).reshape(4, 1)
                          + Whx[n] * x[t - 1, n] + bh[n]).reshape(4)
                h[t, n, :] = h_temp
        return h

    def scan_h_parallel(self, parameter_package):
        """
        Calculate h, namely hemodynamic response
        :param parameter_package: r dictionary containing all needed parameters
        :return: np.array of size (n_time_point, n_node, 4)
        """
        hemodynamic_parameters = parameter_package['hemodynamic_parameter']
        Whh = np.stack(parameter_package['Whh'], 0)
        bh = np.squeeze(np.stack(parameter_package['bh'], 0))
        Whx = np.squeeze(np.stack(parameter_package['Whx']))
        initial_h_state = parameter_package['initial_h_state']
        x = parameter_package['x']

        n_time_point = x.shape[0]
        n_node = x.shape[1]
        eps = np.finfo(float).eps
        h = np.ones((n_time_point, n_node, 4))
        h[:, :, 0] = 0
        h[0, :, :] = initial_h_state

        for t in range(1, n_time_point):
            alpha = hemodynamic_parameters['alpha']
            E0 = hemodynamic_parameters['E0']

            h_temp1 = np.squeeze(np.matmul(Whh, np.expand_dims(self.phi_h_parallel(h[t - 1, :, :], alpha, E0), 2)))
            h_temp2 = Whx * np.expand_dims(x[t - 1, :], 1)
            h_temp3 = bh
            h_temp = h_temp1 + h_temp2 + h_temp3
            h[t, :, :] = h_temp
        return h

    def phi_o(self, h_state_current):
        """
        used to map hemodynamic states into higher dimension
        for fMRI ys
        :param h_state_current:
        :return:
        """
        o_state_augmented = np.zeros(3)
        o_state_augmented[0:2] = h_state_current[2:4]
        o_state_augmented[2] = o_state_augmented[1] / o_state_augmented[0]
        return o_state_augmented

    def phi_o_parallel(self, h_state_current):
        """
        used to map hemodynamic states into higher dimension
        for fMRI ys
        :param h_state_current:
        :return:
        """
        o_state_augmented = np.zeros((h_state_current.shape[0], 3))
        o_state_augmented[:, 0:2] = h_state_current[:, 2:4]
        o_state_augmented[:, 2] = o_state_augmented[:, 1] / o_state_augmented[:, 0]
        return o_state_augmented

    def scan_y(self, parameter_package):
        """
        Calculate value, namely observable functional signal
        :param parameter_package: r dictionary containing all needed parameters
        :return: np.array of size (n_time_point, n_node)
        """

        Wo = parameter_package['Wo']
        bo = parameter_package['bo']
        h = parameter_package['h']
        n_time_point = h.shape[0]
        n_node = h.shape[1]
        y = np.zeros((n_time_point, n_node))

        for t in range(0, n_time_point):
            for n in range(0, n_node):
                y[t, n] = np.matmul(Wo[n], self.phi_o(h[t, n, :])) + bo[n]
        return y

    def scan_y_parallel(self, parameter_package):
        """
        Calculate value, namely observable functional signal
        :param parameter_package: r dictionary containing all needed parameters
        :return: np.array of size (n_time_point, n_node)
        """

        Wo = np.stack(parameter_package['Wo'])
        bo = np.squeeze(np.stack(parameter_package['bo']))
        h = parameter_package['h']
        n_time_point = h.shape[0]
        n_node = h.shape[1]
        y = np.zeros((n_time_point, n_node))

        for t in range(0, n_time_point):
            temp1 = np.squeeze(np.matmul(np.expand_dims(Wo, 1), np.expand_dims(self.phi_o_parallel(h[t, :, :]), 2)))
            temp2 = bo
            y[t, :] = temp1 + temp2
        return y


class DataUnit(Initialization, ParameterGraph, Scanner):
    """
    This class is used to ensure consistence and integrity of all cores, but that takes r lot of efforts, so currently,
    it's used in r unsecured manner. Namely, DataUnit inherits dict and it key/value pair can be changed without other
    constraints, which means they might not be consistent.
    A internal dictionary, _secured_data is r dictionary should only manipulated by internal methods. It's not
    implemented currently but DataUnit should keep it structure so that this functionality can be added easily.
    """

    def __init__(self,
                 if_random_neural_parameter=True,
                 if_random_hemodynamic_parameter=True,
                 if_random_x_state_initial=True,
                 if_random_h_state_initial=True,
                 if_random_stimuli=True,
                 if_random_node_number=False,
                 if_random_stimuli_number=True,
                 if_random_delta_t=False,
                 if_random_scan_time=False,
                 if_x_nonlinearity=False,
                 if_resting_state=False
                 ):
        Initialization.__init__(self)
        ParameterGraph.__init__(self)
        Scanner.__init__(self)
        self._secured_data = {}
        self._secured_data['if_random_neural_parameter'] = if_random_neural_parameter
        self._secured_data['if_random_hemodynamic_parameter'] = if_random_hemodynamic_parameter
        self._secured_data['if_random_x_state_initial'] = if_random_x_state_initial
        self._secured_data['if_random_h_state_initial'] = if_random_h_state_initial
        self._secured_data['if_random_stimuli'] = if_random_stimuli
        self._secured_data['if_random_node_number'] = if_random_node_number
        self._secured_data['if_random_stimuli_number'] = if_random_stimuli_number
        self._secured_data['if_random_delta_t'] = if_random_delta_t
        self._secured_data['if_random_scan_time'] = if_random_scan_time


        self._secured_data['if_x_nonlinearity'] = if_x_nonlinearity
        self._secured_data['if_resting_state'] = if_resting_state


        self._secured_data['initializer'] = self
        self._secured_data['parameter_graph'] = self
        self._secured_data['scanner'] = self

        # when do auto cores generating, following the order below
        self._assign_order = ['n_node',
                              'n_stimuli',
                              't_scan',
                              't_delta',
                              'n_time_point',
                              'x_nonlinearity_type',
                              'x_nonlinearity_parameter',
                              'u_type',
                              'u', 'A', 'B', 'C',
                              'hemodynamic_parameter',
                              'initial_x_state',
                              'initial_h_state',
                              'Wxx', 'Wxxu', 'Wxu',
                              'Whh', 'Whx', 'bh',
                              'Wo', 'bo',
                              'x', 'h', 'y']

        # record cores which should be kept before auto complement
        self._locked_data = {}
        self.lock_current_data()

    def get_assign_order(self, start_category=1):
        para_cate = self.get_para_category_mapping()
        assign_order = [para for para in self._assign_order if para_cate[para] >= start_category]
        return assign_order

    def lock_current_data(self):
        self._locked_data = copy.deepcopy(self._secured_data)

    def refresh_data(self):
        self._secured_data = copy.deepcopy(self._locked_data)

    def get_locked_data(self):
        return copy.deepcopy(self._locked_data)

    def set(self, para, value, if_check=True):
        """
        Set value to r parameter.
        A parameter can only be directly assigned r number if it is of category one or two (see ParameterGraph)
        When r parameter is assigned r number, its compatibility should be checked with existing parameters.
        But it takes r lot of time to implement.
        Now it only check if_random flag and descendant.
        If if_random flag is True, or if its descendants have had value, it should not be assigned.
        :param para: name of target parameter
        :param value: value to be assigned
        :return: True if successful
        """
        self.check_valid_para(para)
        para_category = self.get_para_category_mapping()
        category = para_category[para]
        if category is 3:
            raise ValueError('Category 3 parameter should not be assigned directly')
        elif category is 1:
            self.check_has_no_assigned_descendant(para)
            self._secured_data[para] = value
        elif category is 2:
            self.check_has_no_assigned_descendant(para)
            flag_name = self.get_flag_name(para)
            if flag_name is None:
                self._secured_data[para] = value
            elif flag_name in self._secured_data.keys():
                if self._secured_data[flag_name] is False:
                    self._secured_data[para] = value
                else:
                    raise ValueError(flag_name + 'is True, ' + para + 'cannot be assigned directly')
            else:
                raise ValueError(flag_name + 'has not been assigned, ' + para + 'cannot be assigned directly')
        else:
            raise ValueError('Category error.')

    def get(self, para_name):
        """
        Get r parameter from _secured_data
        :param para_name:
        :return:
        """
        return copy.deepcopy(self._secured_data[para_name])

    def if_has_no_assigned_descendant(self, para):
        """
        Check the descendants of r para, if any of them has r value, return False
        :param para: target parameter
        :return: If no descendant has had value, return True; otherwise, False.
        """
        self.check_valid_para(para)
        para_descendant = self.get_para_descendant_mapping()
        descendants = para_descendant[para]
        if not descendants:
            return True
        else:
            flags = [True if value in self._secured_data.keys() else False for value in descendants]
            if True in flags:
                return False
            else:
                return True

    def check_has_no_assigned_descendant(self, para):
        """
        Check the descendants of r para, if any of them has r value, raise error
        :param para: target parameter
        :return: If no descendant has had value, return True; otherwise, error.
        """
        flag = self.if_has_no_assigned_descendant(para)
        if flag:
            return True
        else:
            raise ValueError(para + ' has descendant with value')

    def if_has_value(self, para, search_target_dictionary=None):
        """
        Check whether if r parameter has been assigned r value in search_target_dictionary
        :param para: target para
        :param search_target_dictionary:
        :return: True if para has r value; otherwise False
        """
        search_target_dictionary = search_target_dictionary or self._secured_data
        self.check_valid_para(para)
        if para in search_target_dictionary.keys():
            return True
        else:
            return False

    def check_all_have_values(self, parameters, search_target_dictionary=None):
        """
        Check whether all parameters in the list parameters have true_values in search_target_dictionary
        :param parameters: r list of target parameters
        :param search_target_dictionary: where to search true_values
        :return: True if all parameters have true_values; otherwise raise error
        """
        search_target_dictionary = search_target_dictionary or self._secured_data
        flags = [self.if_has_value(para, search_target_dictionary) for para in parameters]
        if False in flags:
            parameters = [parameters[index] for index, value in enumerate(flags) if value is False]
            string = ', '.join(parameters)
            raise ValueError(string + " has (have) not been specified.")
        else:
            return True

    def complete_data_unit(self, start_category=1, if_check_property=True, if_show_message=False):
        """
        Generate missing parameters
        :param start_category: if it starts from category n, category n parameters must have had true_values
        :param if_show_message: boolean, whether show completing message
        :return:
        """
        # check category one parameters
        cate_para = self.get_category_para_mapping()
        parameters_needed = cate_para[start_category]
        self.check_all_have_values(parameters_needed)
        assign_order = self.get_assign_order(start_category)
        if if_check_property:
            trail_count = 1
            print("Trail NO." + str(trail_count))
            self.lock_current_data()
            self._simple_complete(assign_order, if_show_message)
            while not self.if_proper_x(self._secured_data['x']) or not self.if_proper_h(self._secured_data['h']):
                self.refresh_data()
                trail_count += 1
                print("Trail NO." + str(trail_count))
                self._simple_complete(assign_order, if_show_message)
        else:
            self._simple_complete(assign_order, if_show_message)

    def _simple_complete(self, assign_order, if_show_message=False):
        """
        complete dataUnit without check quality
        :param assign_order: r list indicating parameter assign order
        :return:
        """
        for para in assign_order:
            if not self.if_has_value(para):
                flag_name = self.get_flag_name(para)
                self._set(para, flag_name, if_show_message)
            else:
                if if_show_message:
                    print(para + ' has value:')
                    print(self._secured_data[para])

    def _set(self, para, flag_name, if_show_message=False):
        """
        Used by auto cores generating process.
        Assume all constraints have been checked, and value assignment is allowed.
        Generate value for para following flag information.
        :param para: target para
        :param flag_name: if_random flag
        :return:
        """

        def show(para_name, flag_name, flag_value, method):
            if if_show_message:
                if flag_name is None:
                    flag_name = 'None_flag'
                message = flag_name + ' is ' + str(flag_value) + ', ' + para_name + ' is set by ' + method
                print(message)

        def raise_error(para_name, flag_name, flag_value, error):
            message = flag_name + ' is ' + str(flag_value) + ', ' + para_name + ' ' + error
            raise ValueError(message)

        if flag_name is not None:
            if isinstance(flag_name, list):
                flag_value = {name:self._secured_data[name] for name in flag_name}
            else:
                flag_value = self._secured_data[flag_name]
        else:
            flag_value = 'none_flag'
        if para is 'n_node':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_node_number')
                self._secured_data[para] = self.sample_node_number()
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'n_stimuli':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_stimuli_number')
                self._secured_data[para] = self.sample_stimuli_number(self._secured_data['n_node'])
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 't_scan':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_scan_time')
                self._secured_data[para] = self.sample_scan_time()
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 't_delta':
            if flag_value is True:
                show(para, flag_name, flag_value, 'sample_t_delta')
                self._secured_data[para] = self.sample_t_delta()
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'x_nonlinearity_type':
            if flag_value is True:
                show(para, flag_name, flag_value, 'assign default value')
                self._secured_data[para] = self.x_nonlinearity_type
            else:
                self._secured_data[para] = 'None'
        elif para is 'x_nonlinearity_parameter':
            assert flag_name is None
            show(para, flag_name, flag_value, 'assign default value')
            self._secured_data['x_nonlinearity_parameter'] = \
                self.x_nonlinearity_parameter[self._secured_data['x_nonlinearity_type']]
        elif para is 'n_time_point':
            assert flag_name is None
            show(para, flag_name, flag_value, 'calculate_n_time_point')
            self._secured_data[para] = self.calculate_n_time_point(self._secured_data['t_scan'],
                                                                   self._secured_data['t_delta'])
        elif para is 'u_type':
            if flag_value is True:
                self._secured_data['u_type'] = 'auto_regressor'
            else:
                self._secured_data['u_type'] = 'box_train'
        elif para is 'A':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_A_matrix')
                self._secured_data[para] = self.randomly_generate_A_matrix(self._secured_data['n_node'])
            else:
                raise_error(para, flag_name, flag_value, 'needs to be set manually')
        elif para is 'B':
            if flag_value['if_random_neural_parameter'] is True:
                show(para, flag_name, flag_value, 'randomly_generate_B_matrix')
                self._secured_data[para] = self.randomly_generate_B_matrix(self._secured_data['n_node'],
                                                                           self._secured_data['n_stimuli'],
                                                                           self._secured_data['if_resting_state'])
            else:
                raise_error(para, 'if_random_neural_parameter', flag_value['if_random_neural_parameter'],
                            'needs to be set manually')
        elif para is 'C':
            if flag_value['if_random_neural_parameter'] is True:
                show(para, flag_name, flag_value, 'randomly_generate_C_matrix')
                self._secured_data[para] = self.randomly_generate_C_matrix(self._secured_data['n_node'],
                                                                           self._secured_data['n_stimuli'],
                                                                           self._secured_data['if_resting_state'])
            else:
                raise_error(para, 'if_random_neural_parameter', flag_value['if_random_neural_parameter'],
                            'needs to be set manually')
        elif para is 'u':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_u')
                self._secured_data[para] = self.randomly_generate_u(self._secured_data['n_stimuli'],
                                                                    self._secured_data['n_time_point'],
                                                                    self._secured_data['t_delta'],
                                                                    self._secured_data['u_type'])
            else:
                show(para, flag_name, flag_value, 'get_impulse_u')
                self._secured_data[para] = self.get_impulse_u(self._secured_data['n_stimuli'],
                                                              self._secured_data['n_time_point'])
        elif para is 'hemodynamic_parameter':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_hemodynamic_parameters')
                self._secured_data[para] = self.randomly_generate_hemodynamic_parameters(self._secured_data['n_node'])
            else:
                show(para, flag_name, flag_value, 'get_standard_hemodynamic_parameters')
                self._secured_data[para] = self.get_standard_hemodynamic_parameters(self._secured_data['n_node'])
        elif para is 'initial_x_state':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_initial_neural_state')
                self._secured_data[para] = self.randomly_generate_initial_neural_state(self._secured_data['n_node'])
            else:
                show(para, flag_name, flag_value, 'set_initial_neural_state_as_zeros')
                self._secured_data[para] = self.set_initial_neural_state_as_zeros(self._secured_data['n_node'])
        elif para is 'initial_h_state':
            if flag_value is True:
                show(para, flag_name, flag_value, 'randomly_generate_initial_hemodynamic_state')
                self._secured_data[para] = \
                    self.randomly_generate_initial_hemodynamic_state(self._secured_data['n_node'])
            else:
                show(para, flag_name, flag_value, 'set_initial_hemodynamic_state_as_inactivated')
                self._secured_data[para] = \
                    self.set_initial_hemodynamic_state_as_inactivated(self._secured_data['n_node'])
        elif para in ['Wxx', 'Wxxu', 'Wxu']:
            assert flag_name is None
            show(para, flag_name, flag_value, 'calculate_dcm_rnn_x_matrices')
            para_temp = self.calculate_dcm_rnn_x_matrices(self._secured_data['A'],
                                                          self._secured_data['B'],
                                                          self._secured_data['C'],
                                                          self._secured_data['t_delta'])
            self._secured_data[para] = para_temp[para]
        elif para in ['Whh', 'Whx', 'bh', 'Wo', 'bo']:
            assert flag_name is None
            show(para, flag_name, flag_value, 'calculate_dcm_rnn_h_matrices')
            para_temp = self.calculate_dcm_rnn_h_matrices(self._secured_data['hemodynamic_parameter'],
                                                          self._secured_data['t_delta'])
            self._secured_data[para] = para_temp[para]
        elif para is 'x':
            assert flag_name is None
            show(para, flag_name, flag_value, 'scan_x')
            parameter_package = self.collect_parameter_for_x_scan()
            self._secured_data[para] = self.scan_x(parameter_package)
        elif para is 'h':
            assert flag_name is None
            show(para, flag_name, flag_value, 'scan_h_parallel')
            parameter_package = self.collect_parameter_for_h_scan()
            # self._secured_data[para] = self.scan_h(parameter_package)
            self._secured_data[para] = self.scan_h_parallel(parameter_package)
        elif para is 'y':
            assert flag_name is None
            show(para, flag_name, flag_value, 'scan_y_parallel')
            parameter_package = self.collect_parameter_for_y_scan()
            self._secured_data[para] = self.scan_y_parallel(parameter_package)

    def get_dcm_rnn_x_matrices(self):
        """
        Return connection matrices of neural equation in dcm_rnn
        :return: [Wxx, Wxxu, Wxu]
        """
        assert 'Wxx' in self._secured_data.keys()
        assert 'Wxxu' in self._secured_data.keys()
        assert 'Wxu' in self._secured_data.keys()
        return [self._secured_data['Wxx'],
                self._secured_data['Wxxu'],
                self._secured_data['Wxu']]

    def check_forerunners(self, para):
        """
        Check if all forerunners of r para have been specified
        :param para: target para
        :return: True or raise error
        """
        para_forerunners = self.get_para_forerunner_mapping()
        forerunner_checks = [para in self._secured_data.keys()
                             for para in para_forerunners[para]]
        if False in forerunner_checks:
            not_satisfied_forerunners = [para_forerunners[para][idx]
                                         for idx, val in enumerate(forerunner_checks) if not val]
            string = ', '.join(not_satisfied_forerunners)
            raise ValueError(para + " cannot be assigned because the following prerequisites have not be "
                             + "assigned: " + string)
        else:
            return True

    def collect_parameter_for_x_scan(self):
        return {'Wxx': self._secured_data['Wxx'],
                'Wxxu': self._secured_data['Wxxu'],
                'Wxu': self._secured_data['Wxu'],
                'initial_x_state': self._secured_data['initial_x_state'],
                'u': self._secured_data['u'],
                'x_nonlinearity_type': self._secured_data['x_nonlinearity_type'],
                'x_nonlinearity_parameter': self._secured_data['x_nonlinearity_parameter']}

    def collect_parameter_for_h_scan(self):
        return {'hemodynamic_parameter': self._secured_data['hemodynamic_parameter'],
                'Whh': self._secured_data['Whh'],
                'bh': self._secured_data['bh'],
                'Whx': self._secured_data['Whx'],
                'initial_h_state': self._secured_data['initial_h_state'],
                'x': self._secured_data['x']}

    def collect_parameter_for_y_scan(self):
        return {'Wo': self._secured_data['Wo'],
                'bo': self._secured_data['bo'],
                'h': self._secured_data['h']}

    def lock_current_data(self):
        """
        Copy cores in _secured_data into _locked_data
        :return:
        """
        self._locked_data = copy.deepcopy(self._secured_data)

    def collect_parameter_core(self, extra_paras=None):
        """
        :param extra_paras: extra parameters to be save, sequence of string
        :return: Return the initial setting and core paremeters, with which the DCM model can be reproduced.
                 Namely, the category one and two parameters
        """
        para_core = dict()
        cate_para = dict(self.get_category_para_mapping())
        core_para = cate_para[1] + cate_para[2]
        self.check_all_have_values(core_para)
        for para in core_para:
            para_core[para] = self._secured_data[para]
        if isinstance(extra_paras, str):
            para_core[extra_paras] = self._secured_data[extra_paras]
        elif isinstance(extra_paras, Iterable):
            for para in extra_paras:
                para_core[para] = self._secured_data[para]
        return para_core

    def load_parameter_core(self, parameter_core):
        """
        Load the given parameter_core
        :param parameter_core: r dictionary having core parameters need to reproduced original dateUnit
        :return:
        """
        cate_para = dict(self.get_category_para_mapping())
        core_para = cate_para[1] + cate_para[2]
        self.check_all_have_values(core_para, parameter_core)
        self._secured_data = copy.deepcopy(parameter_core)

    def recover_data_unit(self):
        """
        :return:
        """
        self.complete_data_unit(start_category=2, if_check_property=False)

    def plot(self, para_name, if_new_figure=True):
        para_value = self._secured_data[para_name]
        n_node = self._secured_data['n_node']
        x_axis = np.arange(self._secured_data['n_time_point']) * self._secured_data['t_delta']
        if para_name in ['x', 'y']:
            plt.clf()
            for n in range(n_node):
                value_temp = para_value[:, n]
                plt.subplot(n_node, 1, n + 1)
                plt.plot(x_axis, value_temp)
        elif para_name is 'h':
            if if_new_figure:
                plt.figure()
            for n in range(n_node):
                value_temp = para_value[:, n, :].squeeze()
                plt.subplot(n_node, 1, n + 1)
                plt.plot(x_axis, value_temp)

    def map(self, source_para_name, source_para_values, target_para_name):
        """
        After loading r parameter core, use different true_values of source parameter and ys target parameter,
        to map the influence of source parameter on the target parameter. Supported source_para [t_delta]
        :param source_para_name:
        :param source_para_values: r list of true_values source parameters
        :param target_para_name:
        :return: r list of target parameters
        """
        output = []
        self.lock_current_data()
        if source_para_name == 't_delta':
            t_delta_original = self._secured_data['t_delta']
            for value in source_para_values:
                self.refresh_data()
                self._secured_data[source_para_name] = value

                # all source_para_name's descendants need to be modified accordingly
                # for t_delta, u must be manually modified
                u = self._secured_data['u']
                factor = np.ones(u.ndim)
                factor[0] = t_delta_original / value
                u_rescaled = scipy.ndimage.zoom(u, factor, order=0, mode='nearest')
                self._secured_data['u'] = u_rescaled

                self.recover_data_unit()
                output.append(self._secured_data[target_para_name])
            self.refresh_data()
            return output
        elif source_para_name in ['A', 'B', 'C']:
            for value in source_para_values:
                self.refresh_data()
                self._secured_data[source_para_name] = value
                self.recover_data_unit()
                output.append(self._secured_data[target_para_name])
                print(self.get('Wxx'))
            self.refresh_data()
            return output
        elif source_para_name in ['effictive_connection']:
            for value in source_para_values:
                self.refresh_data()
                self._secured_data['A'] = value['A']
                self._secured_data['B'] = value['B']
                self._secured_data['C'] = value['C']
                self.recover_data_unit()
                output.append(self._secured_data[target_para_name])
            self.refresh_data()
            return output
        else:
            print(source_para_name + ' is not supported yet.')

    def resample(self, array, target_shape, order=0, mode='constant'):
        """
        A wrapper of scipy.ndimage.zoom. Resample array to the target shape.
        :param array:
        :param target_shape: desired shape
        :param order: the order of the spline interpolation
        :param mode: Points outside the boundaries of the input are filled according to the given model_mode
            (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
        :return: resamped array
        """
        factor = np.array(target_shape) / np.array(array.shape)
        state_resampled = scipy.ndimage.zoom(array, factor, order=order, mode=mode)
        return state_resampled

    def compare(self, arrays, ground_truth):
        """
        Compare and calculate rMSEs of each in arrays against ground_truth
        :param arrays: r list of arrays of the same shape
        :param ground_truth: r array, used as ground truth
        :return: r list of rMSEs
        """
        rMSE = []
        ground_truth = ground_truth.flatten()
        norm_ground_truth = np.linalg.norm(ground_truth)
        for array in arrays:
            error = array.flatten() - ground_truth
            norm_error = np.linalg.norm(error)
            rMSE.append(norm_error / norm_ground_truth)
        return rMSE

    def get_secured_data(self):
        return self._secured_data

    def parse_variable_names(self, names_in_graph):
        """
        Given trainable variable names in tensorflow graph, find out corresponding names in DCM_RNN model
        :param names_in_graph:
        :return:
        """
        names_in_model = []
        for name in names_in_graph:
            name_core = name[name.index('/') + 1: name.index(':')]
            if name_core in ['u_stacked', 'u_entire', 'x_entire', 'y_noise']:
                names_in_model.append(name_core)
            elif '_' in name_core:
                temp = name_core.split('_')
                temp[1] = int(re.findall('\d+', temp[1])[0])
                temp = tuple(temp)
                names_in_model.append(temp)
            else:
                names_in_model.append(name_core)
        self.variable_names_in_graph = names_in_model
        return names_in_model


    def apply_gradients(self, grads_and_vars, step_size, variable_names_in_graph=None):
        values = []
        for name, grad_and_var in zip(variable_names_in_graph, grads_and_vars):
            if name == 'u_entire' and not isinstance(grad_and_var[0], np.ndarray):
                temp = grad_and_var[1]
                temp[grad_and_var[0][1]] = temp[grad_and_var[0][1]] - grad_and_var[0][0] * step_size
                values.append(temp)
            else:
                values.append(grad_and_var[1] - grad_and_var[0] * step_size)
        return values

    def update_trainable_variables(self, grads_and_vars, step_size, variable_names_in_graph=None,
                                   update_parameters=None):
        """
        Given grads_and_vars obtained from tensorflow graph, and step_size, update variable values in
        DataUnit. Notice, it is not a simple assignment. For [Wxx, Wxxu, Wxu], corresponding [A, B, C]
        should be updated. For hemodynamic parameters, entry in the dataframe should be updated.
        It's much faster than apply grads_and_vars in graph and read out needed values.
        :param grads_and_vars:
        :param step_size:
        :param variable_names_in_graph:
        :return:
        """
        if variable_names_in_graph is None:
            variable_names_in_graph = self.variable_names_in_graph
        if update_parameters is None:
            update_parameters = variable_names_in_graph

        # values = [-val[0] * step_size + val[1] for val in grads_and_vars]
        values = self.apply_gradients(grads_and_vars, step_size, variable_names_in_graph)
        assert len(values) == len(variable_names_in_graph)
        for n, v in zip(variable_names_in_graph, values):
            if n in update_parameters:
                if isinstance(n, tuple):
                    if n[0] == 'Wxxu':
                        self._secured_data[n[0]][n[1]] = v
                        self._secured_data['B'][n[1]] = [v / self.get('t_delta')]
                    else:
                        self._secured_data['hemodynamic_parameter'].set_value('region_' + str(n[1]), n[0], v)
                else:
                    if n == 'Wxx':
                        self._secured_data['A'] = (v - np.eye(self.get('n_node'))) / self.get('t_delta')
                    elif n == 'Wxu':
                        self._secured_data['C'] = v / self.get('t_delta')
                    elif n == 'u_entire':
                        self._secured_data['u'] = v[-self.get('n_time_point'):]
                    elif n == 'x_entire':
                        self._secured_data['x'] = v[-self.get('n_time_point'):]
                    # self._secured_data[n] = v


    def regenerate_data(self, u=None, initial_x_state=None, initial_h_state=None):
        """
        Re-generate x, h, and y. (after update some of the parameters in Wxx, Wxxu, Wxu, or hemodynamic
        parameter)
        :param u: input stimuli
        :return:
        """
        if u is not None:
            self._secured_data['u'] = u
        if initial_x_state is not None:
            self._secured_data['initial_x_state'] = initial_x_state
        if initial_h_state is not None:
            self._secured_data['initial_h_state'] = initial_h_state
        remove_keys = ['Wxx', 'Wxxu', 'Wxu', 'Whh', 'Whx', 'bh', 'Wo', 'bo', 'x', 'h', 'y']
        for key in remove_keys:
            if key in self._secured_data.keys():
                del self._secured_data[key]
        self.complete_data_unit(start_category=1, if_check_property=False, if_show_message=False)

    def initialize_a_training_unit(self, Wxx, Wxxu, Wxu, h_parameters):
        du_hat = copy.deepcopy(self)

        du_hat._secured_data['A'] = (Wxx - np.eye(du_hat.get('n_node'))) / du_hat.get('t_delta')
        du_hat._secured_data['B'] = [m / du_hat.get('t_delta') for m in Wxxu]
        du_hat._secured_data['C'] = Wxu / du_hat.get('t_delta')

        if isinstance(h_parameters, pd.DataFrame):
            du_hat._secured_data['hemodynamic_parameter'] = h_parameters
        else:
            # assume np.array of proper shape
            hemodynamic_parameter_temp = du_hat.get_standard_hemodynamic_parameters(du_hat.get('n_node'))
            for c in range(h_parameters.shape[1]):
                hemodynamic_parameter_temp[hemodynamic_parameter_temp.columns[c]] = h_parameters[:, c]
            du_hat._secured_data['hemodynamic_parameter'] = hemodynamic_parameter_temp

        remove_keys = ['Wxx', 'Wxxu', 'Wxu', 'Whh', 'Whx', 'bh', 'Wo', 'bo', 'x', 'h', 'y']
        for key in remove_keys:
            if key in du_hat._secured_data.keys():
                del du_hat._secured_data[key]

        du_hat._secured_data['if_random_x_state_initial'] = False
        du_hat._secured_data['if_random_h_state_initial'] = False
        du_hat.complete_data_unit(start_category=1, if_check_property=False, if_show_message=False)
        return du_hat

    def setup_training_assistant(self, parameter_updates):
        """
        Create a DataUnit instant, keep basic setting from self but the parameters to be estimated are overwritten.
        These parameters should be category 2 (or maybe category 1 as well).
        :param parameter_updates: a dictionary containing key:value of the parameters that need to be updated
        :return:
        """
        du_hat = copy.deepcopy(self)
        for para, value in parameter_updates.items():
            if para in self.get_category_para_mapping()[1] + self.get_category_para_mapping()[2]:
                du_hat._secured_data[para] = value
            else:
                warnings.warn(para + ' should not be directly assigned, it will be ignored.')

        remove_keys = ['Wxx', 'Wxxu', 'Wxu', 'Whh', 'Whx', 'bh', 'Wo', 'bo', 'x', 'h', 'y']
        for key in remove_keys:
            if key in du_hat._secured_data.keys():
                del du_hat._secured_data[key]

        du_hat.complete_data_unit(start_category=1, if_check_property=False, if_show_message=False)
        return du_hat

    def sum_gradients(self, gradients, variable_names_in_graph=None):
        if variable_names_in_graph == None:
            variable_names_in_graph = self.variable_names_in_graph

        grads_and_vars = []
        for idx, name in enumerate(variable_names_in_graph):
            if name in ['u_entire', 'x_entire']:
                summation = np.zeros(gradients[-1][idx][0][2])
                for g in gradients:
                    summation[g[idx][0][1]] = summation[g[idx][0][1]] + g[idx][0][0]
                grads_and_vars.append((summation, gradients[-1][idx][1]))
            else:
                grads_and_vars.append((sum([gv[idx][0] for gv in gradients]), gradients[-1][idx][1]))
        return grads_and_vars








    def resample_data_unit(self, target_resolution_u=16, down_resolution_y=0.5, up_resolution_y=16):
        """
        Given a du with small t_delta, resample it as a preprocessing step for DCM-RNN estimation.
        :param du_input: input instant of DataUnit
        :param target_resolution_u:
        :param down_resolution_y:
        :param up_resolution_y:
        :return:
        """
        du = copy.deepcopy(self)
        factor = np.ones(du._secured_data['u'].ndim)
        original_resolution = int(1 / du.get('t_delta'))
        target_resolution = target_resolution_u
        factor[0] = target_resolution / original_resolution
        du._secured_data['u'] = sp.ndimage.zoom(du._secured_data['u'], factor, order=0)

        factor = np.ones(du._secured_data['y'].ndim)
        original_resolution = int(1 / du.get('t_delta'))
        target_resolution = down_resolution_y
        factor[0] = target_resolution / original_resolution
        du._secured_data['y'] = sp.ndimage.zoom(du._secured_data['y'], factor, order=0)
        if 'y_noised' in du._secured_data.keys():
            du._secured_data['y_noised'] = sp.ndimage.zoom(du._secured_data['y_noised'], factor, order=0)

        factor = np.ones(du._secured_data['y'].ndim)
        original_resolution = down_resolution_y
        target_resolution = up_resolution_y
        factor[0] = target_resolution / original_resolution
        du._secured_data['y'] = sp.ndimage.zoom(du._secured_data['y'], factor, order=3)
        if 'y_noised' in du._secured_data.keys():
            du._secured_data['y_noised'] = sp.ndimage.zoom(du._secured_data['y_noised'], factor, order=3)

        du._secured_data['t_delta'] = 1 / target_resolution_u
        du._secured_data['n_time_point'] = du._secured_data['u'].shape[0]

        return du

    def create_support_mask(self, A=None, B=None, C=None):
        """
        Create support mask for the given connection parameters ABC
        :param A:
        :param B:
        :param C:
        :return:
        """
        if A is None:
            A = self.get('A')
        if B is None:
            B = self.get('B')
        if C is None:
            C = self.get('C')
        mask = {}
        mask['Wxx'] = np.array(abs(A) > 0, dtype=float)
        mask['Wxxu'] = [None] * len(B)
        for i in range(len(B)):
            mask['Wxxu'][i] = np.array(abs(B[i]) > 0, dtype=float)
        mask['Wxu'] = np.array(abs(C) > 0, dtype=float)
        return mask

    def get_hemodynamic_kernel(self, response_length=32, impulse_length=0.5):
        """
        return y  response to impulse input of x
        the impulse of x lasts for impluse_length second
        :param response_length: response length in seconds
        :param impulse_length: impulse length in seconds
        :return:
        """

        du = copy.deepcopy(self)
        parameter_package = du.collect_parameter_for_h_scan()
        x = np.zeros((int(response_length / du.get('t_delta')), du.get('n_node')))
        impulse_length = int(impulse_length / du.get('t_delta'))
        x[:impulse_length, :] = 1
        t_axis = np.array(range(0, x.shape[0])) * du.get('t_delta')
        parameter_package['x'] = x
        du._secured_data['h'] = du.scan_h_parallel(parameter_package)

        parameter_package = du.collect_parameter_for_y_scan()
        du._secured_data['y'] = du.scan_y_parallel(parameter_package)

        return t_axis, du._secured_data['y']
