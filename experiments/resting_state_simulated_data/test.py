import numpy as np

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

    def set(self, index, data):
        self[self.indices[index]] = data

    def get(self, index=None):
        if index == None:
            return self.data
        else:
            return self[self.indices[index]]


test = ArrayWrapper(np.array([1,2,3,4]), 2)
print(type(test))
print(test)

test.set(0, np.array([10, 20]))

test[1]