from typing import List

import os
import time

from ..datasets.bitbots import BallDatasetHandler
from .ReaderTestImages import TestReader
from .LabelReader import Reader as LblReader
from .ImageBatch import ImageBatch


class DataObjectTestSet:

    def __init__(self, pathlist: List[str],
                 batch_size: int = 1,
                 queue_size: int = 16,
                 img_dim: tuple = None,
                 filter_labels = False):
        """
        constructor

        :param pathlist: a list containing valid paths containing images and labels

        :param batch_size: size of the image data batches that will be loaded
        """
        # init variables
        self._queue_size = queue_size
        self._pathlist = pathlist
        self._batch_size = batch_size
        self._labels = LblReader(self._pathlist, img_dim=img_dim)
        self._images = TestReader(self._pathlist, batch_size=self._batch_size, img_dim=img_dim, filter_labels=filter_labels)

        # load data in RAM
        print("loading test data now...")
        self._test_data_batches = list()
        self._load_test_data()
        print("...done!")


    def _load_test_data(self):
        for im_data, im_paths in self._images.get_test_set_iterator():
            labels = self._labels.get_labels_for_batch(im_paths)
            batch = ImageBatch(im_data, labels)

            self._test_data_batches.append(batch)


    def get_set_img(self):
        """
        returns a set of strings for the whole dataset -> "set/filename"
        """
        return self._labels.get_set_img()


    def get_filelist(self):
        """
        returns a list of the files that are hold
        """
        return self._labels.get_label_dict().keys()


    def get_dataset_size(self):
        """
        returns the size of the dataset -> total number of files
        """
        return len(self._labels.get_label_dict().keys())


    def get_paths(self):
        """
        returns the paths from which the images are loaded
        """
        return self._pathlist


    def get_test_set_iterator(self) -> List[ImageBatch]:
        """
        returns the next batch of images & labels
        """
        for batch in self._test_data_batches:
            yield batch


if __name__ == "__main__":
    set_paths = BallDatasetHandler.TEST.CHALLENGE_2018
    starting = time.time()
    data = DataObjectTestSet(set_paths, batch_size=16, queue_size=16, img_dim=(200, 150))
    print(time.time()-starting)
