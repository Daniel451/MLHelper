import numpy as np
from typing import List


class ImageBatch:
    """
    ImageBatch wraps batch information for traing, i.e.
    storing image and label data as well as making this information
    retrievable with getters.

    The order of data is kept since image data is stored as a np.ndarray and labels in a list.
    """

    def __init__(self, img_data: np.ndarray, labels: List):
        self._img_data = img_data
        self._labels = labels


    def get_data(self) -> np.ndarray:
        """
        returns on complete batch of image data
        [shape for images: (batch_size, height, width, color-channels)]
        :return: np.ndarray
        """
        return self._img_data


    def get_labels(self) -> List:
        """
        returns the labels for the complete batch as a list
        each entry i stores the labels for batch_data[i] returned by get_data()
        :return:
        """
        return self._labels


    def get_label_for_image(self, index: int):
        """
        returns the labels for a single image at given index
        :param index: index of the image
        :return: labels
        """
        return self._labels[index]


    def get_shape(self):
        """
        returns the shape of the image data
        :return: ndarray.shape
        """
        return self._img_data.shape

