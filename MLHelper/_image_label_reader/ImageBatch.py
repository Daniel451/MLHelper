import numpy as np
from typing import List


class ImageBatch:

    def __init__(self, img_data : np.ndarray, labels : List):
        self._img_data = img_data
        self._labels = labels


    def get_data(self) -> np.ndarray:
        return self._img_data


    def get_labels(self) -> List:
        return self._labels


    def get_label_for_image(self, index : int):
        return self._labels[index]


    def get_shape(self):
        return self._img_data.shape
