import os, sys
import math
from typing import List, Dict
from itertools import cycle
import numpy as np
import cv2
import time
from tqdm import tqdm


from .PathImageFinder import ImgFind
from .PathImageFinderFilterLabels import ImgFindFilter


class TestReader:

    def __init__(self, pathlist : List[str],
                 batch_size : int = 1,
                 img_dim : tuple = None,
                 filter_labels = False):
        """
        :param pathlist: a list containing valid paths that hold images
        :param batch_size: size of the image data batches that will be loaded
        :param img_dim: resize images to given dimensions - None by default
        """
        # init variables
        self._pathlist = pathlist
        self._batch_size = batch_size
        self._img_paths = np.zeros(1)
        self._img_dim = img_dim
        self._filter_labels = filter_labels

        # load image paths
        self._loading()


    def get_img_paths(self):
        return self._img_paths


    def get_test_set_iterator(self):
        # setup iterator
        index_steps = range(0, self.get_dataset_size(), self._batch_size)

        for idx in index_steps:
            # extract paths for the next image batch
            img_paths = self._img_paths[idx : idx + self._batch_size]

            buffer = list()
            for path in img_paths:
                img_data = cv2.imread(path)
                if self._img_dim is not None:
                    img_data = cv2.resize(img_data, self._img_dim)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                img_data = img_data.astype(np.float32) / 255.0
                buffer.append(img_data)
            img_batch = np.array(buffer).astype(np.float32)

            yield (img_batch, img_paths)


    def _loading(self):
        # info
        print()
        print("reading data...")
        print()

        # load image paths
        if self._filter_labels:
            self._img_paths = ImgFindFilter.find_pngs(self._pathlist)
        else:
            self._img_paths = ImgFind.find_pngs(self._pathlist)
        self._dataset_size = self._img_paths.size

        # summary
        print()
        print("[total number of images]")
        print("{:.>24,}".format(self.get_dataset_size()))
        print()
        print("[filepath array shape {} | array size {:,} KB]".format(self._img_paths.shape, self._img_paths.nbytes / 1000))
        print()


    def get_dataset_size(self):
        """
        returns the number of images (all images)
        """
        return self._dataset_size


    def get_batch_size(self):
        return self._batch_size


    def get_num_of_batches(self):
        return math.ceil(self._dataset_size / self._batch_size)



if __name__ == "__main__":
    sets = ["test-nagoya-game-02"]
    paths = [os.environ["ROBO_AI_DATA"] + iset for iset in sets]

    for batch_size in [1, 2, 4, 8, 10, 16, 20, 32]:
        print()
        print()
        print("###############################")
        print(f"### run with batch size {batch_size:0>3} ###")
        print("###############################")
        rdr = TestReader(paths, batch_size=batch_size)

        path_buffer = list()
        batch_size_buffer = list()
        for im_data, im_paths in tqdm(rdr.get_test_set_iterator(), total=rdr.get_num_of_batches()):
            path_buffer.extend(im_paths)
            batch_size_buffer.append(len(im_paths))

        print()
        batch_min = np.min(batch_size_buffer)
        batch_max = np.max(batch_size_buffer)
        print("number of batches", rdr.get_num_of_batches())
        print("batch size", rdr.get_batch_size())
        print("mean batch size", np.mean(batch_size_buffer))
        print("min batch size", batch_min, " || # of batch min", np.count_nonzero(batch_size_buffer == batch_min))
        print("max batch size", batch_max, " || # of batch max", np.count_nonzero(batch_size_buffer == batch_max))
        print("99th percentile", np.percentile(batch_size_buffer, 99))

        print()
        print("path buffer len", len(path_buffer),
              "test set paths len", len(rdr.get_img_paths()),
              "test set dataset size", rdr.get_dataset_size())

        path_buffer_set = set(path_buffer)
        rdr_set = set(rdr.get_img_paths())

        print()
        print("path buffer set len", len(path_buffer_set),
              "rdr set len", len(rdr_set))

        print()
        print("symmetric difference")
        for diff in sorted(rdr_set.symmetric_difference(path_buffer)):
            print(diff)





