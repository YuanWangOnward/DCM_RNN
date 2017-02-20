import numpy as np
import tensorflow as tf
import math as mth
from scipy.interpolate import interp1d


class get_a_data_helper:
    def __init__(self):
        self.batch_size = 1
        self.n_recurrence = 8
        self.learning_rate = 0.1
        self.stored_data = {}

    def split_data(self, data, truncation_dimension=None, n_recurrence=None, store_name=None):
        # truncate cores into [batch_size, n_recurrent] for training
        # cores are assumed to be np.array like
        # if a cores name is given, it's stored in the class instance with given name
        # the function return a list, each is a training batch
        truncation_dimension = truncation_dimension or data.ndim - 1
        n_recurrence = n_recurrence or self.n_recurrence

        len_total = data.shape[truncation_dimension]

        if len_total % n_recurrence == 0:
            output = np.array_split(data, int(len_total / n_recurrence), truncation_dimension)
        else:
            len_truncated = np.floor(data.shape[truncation_dimension] / n_recurrence) * n_recurrence
            data_truncated = data[:len_truncated]
            output = np.array_split(data_truncated, int(data_truncated / n_recurrence), truncation_dimension)
        if store_name != None:
            self.stored_data[store_name] = output[:]
        return output

    def cut2shape(self, data, shape_target, truncation_dimension=None, n_recurrence=None, store_name=None):
        # truncate cores into [batch_size, n_recurrent] for training
        # cores are assumed to be np.array like
        # if a cores name is given, it's stored in the class instance with given name
        # the function return a list, each is a training batch
        truncation_dimension = truncation_dimension or data.ndim - 1
        n_recurrence = n_recurrence or self.n_recurrence

        len_total = data.shape[truncation_dimension]

        if len_total % n_recurrence == 0:
            output = np.array_split(data, int(len_total / n_recurrence), truncation_dimension)
            # print(output[0].shape)
            # print()
            output = [item.reshape(shape_target) for item in output]
        else:
            len_truncated = np.floor(data.shape[truncation_dimension] / n_recurrence) * n_recurrence
            data_truncated = data[:len_truncated]
            output = np.array_split(data_truncated, int(data_truncated / n_recurrence), truncation_dimension)
            output = [item.reshape(shape_target) for item in output]
        if store_name != None:
            self.stored_data[store_name] = output[:]
        return output
