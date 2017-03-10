from unittest import TestCase
import numpy as np
import DCM_RNN.toolboxes as tb

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


