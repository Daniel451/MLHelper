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

        # metrics -- prediction vs. ground truth
        self.TP = 4
        self.TN = 9
        self.FP = 1
        self.FN = 2
        self.intersection = 4.0
        self.union = 7.0

        # confusion matrix
        self.confusion_matrix = np.array([
            [self.TP, self.FP],
            [self.FN, self.TN]
        ]).astype(np.int)

    def test_threshold_binary(self):
        bin_pred = H.heatmap.threshold_binary(heatmap=self.heatmap_pred, threshold=0.5)
        bin_true = H.heatmap.threshold_binary(heatmap=self.heatmap_true, threshold=0.5)

        self.assertTrue(np.array_equal(bin_pred, self.binary_pred))
        self.assertTrue(np.array_equal(bin_true, self.binary_true))

    def test_metrics_tp_tn_fp_fn(self):
        res = H.heatmap.metrics_tp_tn_fp_fn(prediction=self.binary_pred, ground_truth=self.binary_true)

        self.assertEqual(res.TP, self.TP)
        self.assertEqual(res.TN, self.TN)
        self.assertEqual(res.FP, self.FP)
        self.assertEqual(res.FN, self.FN)
        self.assertEqual(res.ALL, 16)

    def test_intersection_over_union(self):
        iou = H.heatmap.intersection_over_union(heatmap1=self.binary_pred, heatmap2=self.binary_true)

        self.assertEqual(iou, self.intersection/self.union)

    def test_confusion_matrix(self):
        cfm = H.heatmap.confusion_matrix(prediction=self.binary_pred, ground_truth=self.binary_true)

        self.assertTrue(np.array_equal(cfm, self.confusion_matrix))

    def test_heatmap_labels2D_rectangular(self):
        # set up dummy data
        heatmap = np.zeros(shape=(2, 10, 10, 1), dtype=np.float32)
        heatmap[0, 1:4, 1:4, :] = 1.0
        heatmap[1, 5:10, 7:10, :] = 1.0

        # set up dummy labels
        bbx1 = H.LabelBoundingBox2D(
            x1=1,
            y1=1,
            x2=3,
            y2=3
        )

        bbx2 = H.LabelBoundingBox2D(
            x1=7,
            y1=5,
            x2=9,
            y2=9
        )

        labels = [[bbx1], [bbx2]]

        ret = H.heatmap.labels2D_rectangular(heatmap, labels)

        self.assertTrue(np.array_equal(heatmap, ret))







