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
