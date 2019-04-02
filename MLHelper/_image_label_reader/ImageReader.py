import os, sys
from typing import List, Dict
from itertools import cycle
import numpy as np
import cv2
import multiprocessing as mp
import time

from .PathImageFinder import ImgFind
from .PathImageFinderFilterLabels import ImgFindFilter


class Reader:
    def __init__(self, pathlist : List[str],
                 label_content: str = None,
                 batch_size : int = 1,
                 queue_size : int = 16,
                 processes : int = None,
                 img_dim : tuple = (200, 150),
                 wait_for_queue_full = True,
                 filter_labels = False):
        """
        :param pathlist: a list containing valid paths that hold images
        :param batch_size: size of the image data batches that will be loaded
        :param img_dim: resize images to given dimensions - None by default
        :param wait_for_queue_full: toggles if main thread should wait for the queue to be full before continuing
        :param processes: number of processes to start as workers for loading images
        """
        # init variables
        self._pathlist = pathlist
        self._batch_size = batch_size
        self._img_paths = np.zeros(1)
        self._file_cycler = None
        self._img_dim = img_dim
        self._queue_size = queue_size
        self._filter_labels = filter_labels

        # data queue
        self._q = mp.Queue(self._queue_size)
        self._index_q = mp.Queue(20)

        # set label
        self._label_content = label_content

        # load image paths
        self._loading()

        # setup iterator
        self._random_img_paths = self._img_paths.copy()
        np.random.shuffle(self._random_img_paths)
        random_index_steps = range(0, self.get_dataset_size(), self._batch_size)
        self._file_cycling_index = cycle(random_index_steps)

        # index queue worker
        worker = mp.Process(target=self._index_worker)
        worker.daemon = True
        worker.start()

        # determine appropriate number of processes
        if processes is None:
            try:
                num_workers = int(mp.cpu_count() / 2)
            except Exception:
                num_workers = 4
        else:
            num_workers = processes

        # start multiprocessing worker for filling queue
        for i in range(num_workers):
            worker = mp.Process(target=self._worker)
            worker.daemon = True
            worker.start()

        # potentially wait in the main thread for the queue to be full once before continuing
        if wait_for_queue_full:
            wait = 0
            while not self._q.full():
                time.sleep(1)
                wait += 1
                if wait > 7:
                    print(f"[{self._q.qsize()}/{self.get_max_queue_size()} batches] image queue is still loading...")
        print()


    def _index_worker(self):
        while True:
            next_index = next(self._file_cycling_index)
            self._index_q.put(next_index)


    def get_max_queue_size(self):
        return self._queue_size


    def get_next_img_batch(self) -> (np.ndarray, list):
        """
        returns the next image batch in queue and the corresponding filepaths

        (np.ndarray, list) -> (img_data, img_filepaths)
        """
        return self._q.get()


    def _worker(self):
        while True:
            # get next index delimiter
            next_index = self._index_q.get(block=True)

            # extract paths for the next image batch
            img_paths = self._random_img_paths[next_index: next_index + self._batch_size]

            buffer = list()
            for path in img_paths:
                img_data = cv2.imread(path)
                assert img_data is not None, \
                    f"img_data is None, OpenCV could not read '{path}'"
                if self._img_dim is not None:
                    img_data = cv2.resize(img_data, self._img_dim)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                img_data = img_data.astype(np.float32) / 255.0
                assert (img_data.ndim == 2 or img_data.ndim == 3), \
                    f"img_data.ndim was '{img_data.ndim}', which is unsupported for images"
                buffer.append(img_data)
            img_batch = np.array(buffer).astype(np.float32)

            # fill queue
            self._q.put((img_batch, img_paths), block=True)


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
