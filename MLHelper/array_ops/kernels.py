import numpy as np


def noise_kernel(size):
    """
    create a kernel matrix for image noising

    :param size: dimensionality (size x size)
    """
    k = np.zeros((size, size))
    k[int((size - 1) / 2), :] = np.ones(size)
    k = k / size

    if k.sum() > 1.0:
        eps = k.sum() - 1.0
        k = np.maximum(k - eps, 0.0)
        k = k / k.sum()

    if k.min() < 0.0:
        eps = np.abs(k.min())
        k = np.minimum(k + eps, 1.0)
        k = k / k.sum()

    return k
