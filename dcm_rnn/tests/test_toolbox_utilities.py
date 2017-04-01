from unittest import TestCase
import numpy as np
import dcm_rnn.toolboxes as tb

class TestToolboxUtilities(TestCase):
    def test_split(self):
        n_segment = 4
        n_step = 4
        shift = 0
        split_dimension = 0
        data = np.arange(0, 12)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        np.testing.assert_array_equal(splits[0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(splits[1], np.array([4, 5, 6, 7]))
        np.testing.assert_array_equal(splits[2], np.array([8, 9, 10, 11]))

        n_segment = 4
        n_step = 2
        shift = 0
        split_dimension = 0
        data = np.arange(0, 8)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        np.testing.assert_array_equal(splits[0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(splits[1], np.array([2, 3, 4, 5]))
        np.testing.assert_array_equal(splits[2], np.array([4, 5, 6, 7]))

        n_segment = 4
        n_step = 2
        shift = 0
        split_dimension = 0
        data = np.random.rand(8, 40, 40)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        np.testing.assert_array_equal(splits[0], np.take(data, range(0, 4), split_dimension))
        np.testing.assert_array_equal(splits[1], np.take(data, range(2, 6), split_dimension))
        np.testing.assert_array_equal(splits[2], np.take(data, range(4, 8), split_dimension))

        n_segment = 4
        n_step = 2
        shift = 0
        split_dimension = 1
        data = np.random.rand(40, 8, 40)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        np.testing.assert_array_equal(splits[0], np.take(data, range(0, 4), split_dimension))
        np.testing.assert_array_equal(splits[1], np.take(data, range(2, 6), split_dimension))
        np.testing.assert_array_equal(splits[2], np.take(data, range(4, 8), split_dimension))

        n_segment = 4
        n_step = 2
        shift = 2
        split_dimension = 1
        data = np.random.rand(40, 8, 40)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        np.testing.assert_array_equal(splits[0], np.take(data, range(2, 6), split_dimension))
        np.testing.assert_array_equal(splits[1], np.take(data, range(4, 8), split_dimension))

    def test_merge(self):
        n_segment = 4
        n_step = 4
        shift = 0
        split_dimension = 0
        data = np.random.rand(16)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        merged = tb.merge(splits, n_segment=n_segment, n_step=n_step, merge_dimension=split_dimension)
        np.testing.assert_array_equal(data, merged)

        n_segment = 4
        n_step = 2
        shift = 0
        split_dimension = 0
        data = np.random.rand(16)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        merged = tb.merge(splits, n_segment=n_segment, n_step=n_step, merge_dimension=split_dimension)
        np.testing.assert_array_equal(data, merged)

        n_segment = 4
        n_step = 2
        shift = 2
        split_dimension = 1
        data = np.random.rand(40, 8, 40)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        merged = tb.merge(splits, n_segment=n_segment, n_step=n_step, merge_dimension=split_dimension)
        np.testing.assert_array_equal(np.take(data, range(shift, 8), split_dimension), merged)

        n_segment = 4
        n_step = 2
        shift = 3
        split_dimension = 0
        data = np.random.rand(40, 8, 40)
        splits = tb.split(data, n_segment=n_segment, n_step=n_step, shift=shift, split_dimension=split_dimension)
        merged = tb.merge(splits, n_segment=n_segment, n_step=n_step, merge_dimension=split_dimension)
        np.testing.assert_array_equal(
            np.take(data, range(shift, shift + merged.shape[split_dimension]), split_dimension), merged)

    def test_split_index(self):
        # split_index(data_shape, n_segment, n_step=None, shift=0, split_dimension=0)
        array = np.ones((2, 8, 10))
        indices = tb.split_index(array.shape, n_segment=2)
        array[indices[0]] = 0
        array_true = np.ones((2, 8, 10))
        array_true[0: 2, :, :] = 0
        np.testing.assert_array_equal(array, array_true)


        array = np.ones((8, 8, 10))
        indices = tb.split_index(array.shape, n_segment=2, n_step=1)
        array[indices[0]] = 0
        array[indices[1]] = 2
        array_true = np.ones((8, 8, 10))
        array_true[0: 2, :] = 0
        array_true[1: 3, :] = 2
        np.testing.assert_array_equal(array, array_true)

        array = np.ones((8, 8, 10))
        indices = tb.split_index(array.shape, n_segment=2, n_step=1, split_dimension=1)
        array[indices[0]] = 0
        array[indices[1]] = 2
        array_true = np.ones((8, 8, 10))
        array_true[:, 0: 2, :] = 0
        array_true[:, 1: 3, :] = 2
        np.testing.assert_array_equal(array, array_true)

        array = np.ones((8, 8, 10))
        indices = tb.split_index(array.shape, n_segment=4, n_step=2, shift=1, split_dimension=1)
        array[indices[0]] = 0
        array[indices[1]] = 2
        array_true = np.ones((8, 8, 10))
        array_true[:, 1: 5, :] = 0
        array_true[:, 3: 7, :] = 2
        np.testing.assert_array_equal(array, array_true)







