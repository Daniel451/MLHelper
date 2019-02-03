import numpy as np



def threshold_binary(heatmap: np.ndarray, threshold: float) -> np.ndarray:
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
