import os, sys
from typing import List, Dict, Union
from itertools import cycle
import numpy as np
import cv2
import multiprocessing as mp
import time
import math

from .PathImageFinder import ImgFind
from .PathImageFinderFilterLabels import ImgFindFilter
from ..datasets.bitbots import ImagesetCollection


class Reader:
    def __init__(self, collection_or_paths: Union[ImagesetCollection, List[str]],
                 label_content: str = None,
                 batch_size: int = 1,
                 queue_size: int = 16,
                 processes: int = None,
                 img_dim: tuple = (200, 150),
                 wait_for_queue_full=True,
                 filter_labels=False):
        """
        :param collection_or_paths: an ImagesetCollection instance or a list containing valid paths that hold images
        :param batch_size: size of the image data batches that will be loaded
        :param img_dim: resize images to given dimensions - None by default
        :param wait_for_queue_full: toggles if main thread should wait for the queue to be full before continuing
        :param processes: number of processes to start as workers for loading images
        """
        # init variables
        if isinstance(collection_or_paths, ImagesetCollection):
            self._pathlist = collection_or_paths.to_paths()
        else:
            self._pathlist = collection_or_paths
        self._batch_size = batch_size
        self._img_paths = np.zeros(1)
        self._file_cycler = None
        self._img_dim = img_dim
        self._queue_size = queue_size
        self._filter_labels = filter_labels
        self._dataset_size = 0
        self._label_content = label_content

        # determine appropriate number of processes
        if processes is None:
            try:
                num_workers = int(mp.cpu_count() / 2)
            except Exception:
                num_workers = 4
        else:
            num_workers = processes
        print(f"MLHelper ImageReader working with {num_workers} processes for image loading")

        # load image paths
        self._loading()

        # data queue
        self._q = mp.Queue(self._queue_size)  # image queue size, i.e. max. number of batches to store in image queue
        self._imgpath_q = mp.Queue(int(num_workers*32))  # statically setting queue size much bigger than num_workers

        # index queue worker
        path_worker = mp.Process(target=self._path_chunk_worker,
                                 args=(self._imgpath_q, self._img_paths,))
        path_worker.daemon = True
        path_worker.start()

        # start multiprocessing worker for filling queue
        img_worker = list()
        for i in range(num_workers):
            worker = mp.Process(target=self._worker,
                                args=(self._imgpath_q, self._q))
            worker.daemon = True
            worker.start()
            img_worker.append(worker)

        # potentially wait in the main thread for the queue to be full once before continuing
        if wait_for_queue_full:
            wait = 0
            while not self._q.full():
                time.sleep(1)
                wait += 1
                if wait > 7:
                    print(f"[{self._q.qsize()}/{self.get_max_queue_size()} batches] image queue is still loading...")
        print()


    def _path_chunk_worker(self, path_q: mp.Queue, imgpaths: np.ndarray):
        # helper
        def get_random_paths():
            imgpaths_random = imgpaths.copy()
            np.random.shuffle(imgpaths_random)
            imgpaths_random = np.concatenate((imgpaths_random, imgpaths_random[0:self._batch_size]), axis=0)

            return imgpaths_random

        # loop
        while True:
            # get new random paths
            paths = get_random_paths()
            indexes = range(0, self._dataset_size, self._batch_size)

            # batches for one whole epoch
            for idx in indexes:
                # add final batch of paths to the queue
                path_q.put(paths[idx : idx + self._batch_size], block=True)


    def get_max_queue_size(self):
        return self._queue_size


    def get_next_img_batch(self) -> (np.ndarray, list):
        """
        returns the next image batch in queue and the corresponding filepaths

        (np.ndarray, list) -> (img_data, img_filepaths)
        """
        return self._q.get()


    def _worker(self, path_q: mp.Queue, img_q):
        while True:
            # extract paths for the next image batch
            img_paths = path_q.get(block=True)

            buffer = list()
            for path in img_paths:
                # read in image
                img_data = cv2.imread(path)

                # check if reading succeeded
                assert img_data is not None, \
                    f"img_data is None, OpenCV could not read '{path}'"

                # if img_dim is set, resize image
                if self._img_dim is not None:
                    img_data = cv2.resize(img_data, self._img_dim)

                # RGB color correction
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

                # change dtype to float32
                img_data = img_data.astype(np.float32) / 255.0

                # check image ndim
                assert (img_data.ndim == 2 or img_data.ndim == 3), \
                    f"img_data.ndim was '{img_data.ndim}', which is unsupported for images"

                # append image data to buffer
                buffer.append(img_data)

            # convert list to proper numpy array
            img_batch = np.array(buffer).astype(np.float32)

            # fill queue
            img_q.put((img_batch, img_paths), block=True)


    def _loading(self):
        # info
        print()
        print("reading data...")
        print()

        # load image paths
        if self._filter_labels and self._label_content is not None:
            self._img_paths = ImgFindFilter.find_pngs(self._pathlist, self._label_content)
        else:
            self._img_paths = ImgFind.find_pngs(self._pathlist)
        self._dataset_size = self._img_paths.size

        # summary
        print()
        print("[total number of images]")
        print("{:.>24,}".format(self.get_dataset_size()))
        print()
        print("[filepath array shape {} | array size {:,} KB]".format(self._img_paths.shape,
                                                                      self._img_paths.nbytes / 1000))
        print()


    def get_dataset_size(self):
        """
        returns the number of images (all images)
        """
        return self._dataset_size


if __name__ == "__main__":
    sets = ["bitbots-set00-02/", "bitbots-set00-03", "bitbots-set00-04"]
    paths = [os.environ["ROBO_AI_DATA"] + iset for iset in sets]
    label_content = "ball"
    r = Reader(paths, label_content, batch_size=8, queue_size=64)
