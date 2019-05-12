from typing import List, Union

from .ImageReader import Reader as ImgReader
from .LabelReader import Reader as LblReader
from .ImageBatch import ImageBatch
from ..datasets.bitbots import ImagesetCollection


class DataObject:

    def __init__(self, collection_or_paths : Union[ImagesetCollection, List[str]],
                 label_content : str = None,
                 batch_size : int = 1,
                 queue_size : int = 64,
                 img_dim : tuple = (200, 150),
                 processes : int = None,
                 filter_labels = False):
        """
        constructor

        :param collection_or_paths: an ImagesetCollection instance or a list containing valid paths containing images and labels

        :param batch_size: size of the image data batches that will be loaded
        """
        # init variables
        if isinstance(collection_or_paths, ImagesetCollection):
            self._pathlist = collection_or_paths.to_paths()
        else:
            self._pathlist = collection_or_paths
        self._batch_size = batch_size
        self._labels = LblReader(self._pathlist, label_content=label_content, img_dim=img_dim)
        self._images = ImgReader(self._pathlist, label_content=label_content, batch_size=self._batch_size,
                                 queue_size=queue_size, img_dim=img_dim, filter_labels=filter_labels,
                                 processes=processes)


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


    def get_next_batch(self) -> ImageBatch:
        """
        returns the next batch of images & labels
        """
        imgdata, filepaths = self._images.get_next_img_batch()
        labels = self._labels.get_labels_for_batch(filepaths)

        batch = ImageBatch(imgdata, labels)

        return batch
