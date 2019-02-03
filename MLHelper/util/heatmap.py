from collections import namedtuple
import numpy as np



def threshold_binary(heatmap: np.ndarray, threshold: float) -> np.ndarray[bool]:
    """
    Returns a boolean array where each entry is True if the
    corresponding entry in heatmap >= the given threshold.

    :param heatmap: input array
    :param threshold: float threshold
    :return: boolean array
    """
    arr = np.zeros_like(heatmap, dtype=np.bool)
    arr[np.where(heatmap >= threshold)] = True
    return arr


def intersection_over_union(heatmap1: np.ndarray[bool], heatmap2: np.ndarray[bool]) -> float:
    """
    Metric. Returns the intersection over union (jaccard index) for two given heatmaps.
    Heatmaps need to be boolean arrays -> True for true positives, False otherwise.

    :param heatmap1: np.ndarray[bool]
    :param heatmap2: np.ndarray[bool]
    :return: float; intersection over union metric
    """
    intersection = np.bitwise_and(heatmap1, heatmap2)
    union = np.bitwise_or(heatmap1, heatmap2)

    count_inter = float(np.count_nonzero(intersection))
    count_union = float(np.count_nonzero(union))

    iou = count_inter / count_union

    return iou


def metrics_tp_tn_fp_fn(prediction: np.ndarray[bool], ground_truth: np.ndarray[bool]) -> namedtuple:
    """
    Given two boolean heatmaps 'prediction' and 'ground_truth' this function computes the number of
    True Positives [TP], True Negatives [TN], False Positives [FP], False Negatives [FN] and returns
    them as a namedtuple.

    :param prediction: np.ndarray[bool] of predicted heatmap
    :param ground_truth: np.ndarray[bool] of ground truth values
    :return: namedtuple("metrics", ["TP", "TN", "FP", "FN", "ALL"])
    """
    metrics = namedtuple("metrics", ["TP", "TN", "FP", "FN", "ALL"])

    pred_pos = prediction
    pred_neg = np.bitwise_not(prediction)
    true_pos = ground_truth
    true_neg = np.bitwise_not(ground_truth)

    # TP
    TP = np.count_nonzero(np.bitwise_and(pred_pos, true_pos))

    # TN
    TN = np.count_nonzero(np.bitwise_and(pred_neg, true_neg))

    # FP
    FP = np.count_nonzero(np.bitwise_and(pred_pos, true_neg))

    # FN
    FN = np.count_nonzero(np.bitwise_and(pred_neg, true_pos))

    # ALL
    ALL = TP+TN+FP+FN

    return metrics(TP=TP, TN=TN, FP=FP, FN=FN, ALL=ALL)


def confusion_matrix(prediction: np.ndarray[bool], ground_truth: np.ndarray[bool]) -> namedtuple:
    """
    Computes the confusion matrix of two boolean heatmaps 'prediction' and 'ground_truth'.

    :param prediction: np.ndarray[bool] of predicted heatmap
    :param ground_truth: np.ndarray[bool] of ground truth values
    :return:
    """
    # gather metrics
    res = metrics_tp_tn_fp_fn(prediction=prediction, ground_truth=ground_truth)

    cfm = np.array([
        [res.TP, res.FP],
        [res.FN, res.TN]
    ]).astype(np.int)

    return cfm


