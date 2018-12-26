from typing import List
import os
import glob
import numpy as np



class ImgFind(object):

    @staticmethod
    def find_pngs(pathlist : List[str]) -> np.ndarray:
        pathlist = ImgFind.check_path_validity(pathlist)

        images = [ImgFind.worker(path) for path in pathlist]

        # flatten output arrays & sort afterwards
        images = np.concatenate(images, axis=0)
        images = np.sort(images)

        return images


    @staticmethod
    def worker(path : str) -> np.ndarray:
        images = glob.glob(os.path.join(path, "*.png"))
        images.extend(glob.glob(os.path.join(path, "*.jpg")))

        if not len(images) > 0:
            raise Exception("no png files found in path '{}'".format(path))

        print("[{}]".format(path))
        print("> number of images{:.>16,}".format(len(images)))
        print()

        return np.array(images)


    @staticmethod
    def check_path_validity(pathlist : List[str]) -> List[str]:
        """
        checks a list of paths for validity and returns the list
        """
        for path in pathlist:
            # check if path is a str
            if not type(path) == str:
                raise TypeError("path has to be a string but was {}".format(type(path)))

            # check if directory exists
            if not os.path.exists(path):
                raise IOError("specified path '{}' does not exist".format(path))

            # check if directory is valid directory
            if not os.path.isdir(path):
                raise IOError("specified path '{}' is not a valid directory".format(path))

        return pathlist


