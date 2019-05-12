from typing import Iterable
import os
import glob
import math
import time
from typing import List, Dict, Union
from collections import defaultdict
from .LabelObjects import LabelBoundingBox2D
from ..datasets.bitbots import ImagesetCollection


class Reader:
    def __init__(self, collection_or_paths: Union[ImagesetCollection, List[str]],
                 label_content: str,
                 img_dim: tuple = None):
        """
        :param collection_or_paths: an ImagesetCollection instance or a list containing valid paths that hold images
        :param img_dim: resize images to given dimensions - None by default
        """
        # init variables
        if isinstance(collection_or_paths, ImagesetCollection):
            self._pathlist = collection_or_paths.to_paths()
        else:
            self._pathlist = collection_or_paths
        self._label_content = label_content
        self._img_dim = img_dim
        self._labels = defaultdict(list)
        self._set_img = set()

        # check all paths for labels
        self._process_paths()


    def get_labels_for_batch(self, batch_filenames: Iterable[str]) -> List[List]:
        """
        returns the labels for a whole batch of filenames (ordered)
        :param batch_filenames: a list (batch) of filenames/-paths of image files
        """
        buffer = list()

        for filepath in batch_filenames:
            # self._labels contains lists for each filepath encapsulating the actual LabelBoundBox2D objects
            buffer.append(self._labels[filepath])

        return buffer


    def get_set_img(self):
        return self._set_img


    def _process_paths(self):
        for path in self._pathlist:
            txt_filepaths = glob.glob(os.path.join(path, "*.txt"))

            for filepath in txt_filepaths:
                self._process_txt_file(filepath, path)


    def _extract_set_line(self, filestream) -> (bool, str):
        """
        Receives a filestream object (with open()) and searches for imageset information
        in the first lines of given filestream. Returns if imageset information was found,
        including the imageset name.

        :return: (found_set_information, set_name)
        """
        found_set_line = False
        set_name = None

        for line in filestream:
            sline = line.strip()
            if sline.startswith("[set|") and sline.endswith("]"):
                found_set_line = True
                # example: extract the set name 'bitbots-set00-01'
                # out of the whole string "[set|bitbots-set00-01]"
                set_name = sline.split("|")[1]
                set_name = set_name[:-1]
                print(f"[found] information for set '{set_name}'...")
                break

        return found_set_line, set_name


    def _process_txt_file(self, filepath, dirpath):
        """
        process one txt file, i.e. checking for imageset information, extracting labels,
        and storing label information
        """
        with open(filepath, mode="r") as f:
            counter = 0
            print()
            print(f"[{os.path.basename(filepath)}]")
            print(f"[check] checking path '{filepath}' for '{self._label_content}' labels...")

            # check header information
            found_set_line, set_name = self._extract_set_line(f)

            # if format line was not found: return
            if not found_set_line or set_name is None:
                return

            # extract labels line by line
            # starting at the first label-line
            for line in f:
                sline = line.strip()

                # filter not_in_image -> skip iteration
                if "not_in_image" in sline:
                    continue
                elif sline.startswith("label::" + self._label_content):
                    try:
                        label_type, filename, img_width, img_height, \
                        x1, y1, x2, y2, \
                        cx, cy, width, height = sline.split("|")
                    except ValueError as e:
                        msg = "\n"
                        msg += f"exception: {e}"
                        msg += f"error occured for line:\n{line})"
                        exit(msg)

                    # if not filename in self._labels:
                    counter += 1
                    x1 = int(x1.strip())
                    y1 = int(y1.strip())
                    x2 = int(x2.strip())
                    y2 = int(y2.strip())
                    img_width = int(img_width)
                    img_height = int(img_height)

                    # if images are resized dynamically,
                    # coordinates need to be re-calculated
                    # in order to match new image dimensions
                    if self._img_dim is not None:
                        xfactor = img_width / float(self._img_dim[0])
                        yfactor = img_height / float(self._img_dim[1])
                        x1 = math.floor(float(x1) / xfactor)
                        y1 = math.floor(float(y1) / yfactor)
                        x2 = math.ceil(float(x2) / xfactor)
                        y2 = math.ceil(float(y2) / yfactor)

                    self._set_img.add(f"{set_name}/{filename}")

                    # create a namedtuple to store label information
                    bbx = LabelBoundingBox2D(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        image_width=img_width,
                        image_height=img_height,
                        set_name=set_name,
                        filename=filename
                    )

                    # add tuple to label information for current image
                    dict_label_id = os.path.join(dirpath, filename)
                    self._labels[dict_label_id].append(bbx)

            print(f"[read] {counter} '{self._label_content}' labels read for set '{set_name}' from file '{filepath}'...")


    def get_pathlist(self):
        return self._pathlist


    def get_label_dict(self) -> dict:
        return self._labels


if __name__ == "__main__":
    sets = ["bitbots-set00-02/", "bitbots-set00-03", "bitbots-set00-04"]
    paths = [os.environ["ROBO_AI_DATA"] + iset for iset in sets]
    label_content = "ball"
    r = Reader(paths, label_content)
    print(len(r.get_label_dict()))
