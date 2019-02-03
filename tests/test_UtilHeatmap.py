import numpy as np
import unittest
import MLHelper as H


class TestUtilHeatmap(unittest.TestCase):

    def setUp(self):
        # sample - predicted heatmap
        self.heatmap_pred = np.array([
            [0.0, 0.2, 0.6, 0.95],
            [0.0, 0.1, 0.5, 0.92],
            [0.2, 0.2, 0.45, 0.8],
            [0.1, 0.3, 0.4, 0.47]
        ]).astype(np.float32)

        # sample - ground truth heatmap
        self.heatmap_true = np.array([
            [0.0, 0.8, 1.0, 0.8],
            [0.0, 0.6, 0.8, 0.6],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]).astype(np.float32)

        # binary version of prediction -- threshold >= 0.5
        self.binary_pred = np.array([
            [False, False, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False]
        ]).astype(np.bool)

        # binary version of ground truth -- threshold >= 0.5
        self.binary_true = np.array([
            [False, True, True, True],
            [False, True, True, True],
            [False, False, False, False],
            [False, False, False, False]
        ]).astype(np.bool)

    def test_threshold_binary(self):
        bin_pred = H.heatmap.threshold_binary(heatmap=self.heatmap_pred, threshold=0.5)
        bin_true = H.heatmap.threshold_binary(heatmap=self.heatmap_true, threshold=0.5)

        self.assertTrue(np.array_equal(bin_pred, self.binary_pred))
        self.assertTrue(np.array_equal(bin_true, self.binary_true))

