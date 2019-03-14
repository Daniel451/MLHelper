from typing import Iterable
import os
import glob
import math
from typing import List, Dict


class Reader:
    def __init__(self, pathlist: List[str], label_content: str, img_dim: tuple = None):
        """
        :param pathlist: a list containing valid paths that hold images
        :param img_dim: resize images to given dimensions - None by default
        """
        # init variables
        self._pathlist = pathlist
        self._label_content = label_content
        self._img_dim = img_dim
        self._labels = dict()
        self._set_img = set()

        # check all paths for labels
        self._process_paths()


    def get_labels_for_batch(self, batch_filenames: Iterable[str]) -> List[Dict]:
        """
        returns the labels for a whole batch of filenames (sorted)
        :param batch_filenames: a list (batch) of filenames/-paths of image files
        """
        buffer = list()
        #counter = 0
        #for fp in batch_filenames:
        #    #Workaround for stable interface. Need to be changed to return Lists with multiple labels
        #    buffer.append(dict(self._labels[fp]))
        #    print(len(self._labels[fp]))

        for fp in batch_filenames:
            buffer.extend([self._labels[fp]])
        return buffer


    def get_set_img(self):
        return self._set_img


    def _process_paths(self):
        for path in self._pathlist:
            txt_filepaths = glob.glob(os.path.join(path, "*.txt"))

            for filepath in txt_filepaths:
                self._process_txt_file(filepath, path)


    def _process_txt_file(self, filepath, dirpath):
        with open(filepath, mode="r") as f:
            counter = 0
            print()
            print("checking '{}' for labels...".format(filepath))
            found_set_line = False

            for line in f:
                sline = line.strip()
                if sline.startswith("[set|") and sline.endswith("]"):
                    found_set_line = True
                    # example: extract the set name 'bitbots-set00-01'
                    # out of the whole string "[set|bitbots-set00-01]"
                    set_name = sline.split("|")[1]
                    set_name = set_name[:-1]
                    print("found annotations for set '{}'...".format(set_name))
                    break

            # search for the format line 'imagename|x1|y1|x2|y2' line by line
            # labels are saved directly after this line
            # if found_set_line:
            #     for line in f:
            #         if "imagename|x1|y1|x2|y2" in line:
            #             found_format_line = True
            #             break

            # if format line was found, extract labels line by line
            if found_set_line:
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

                        #if not filename in self._labels:
                        counter += 1
                        x1 = int(x1.strip())
                        y1 = int(y1.strip())
                        x2 = int(x2.strip())
                        y2 = int(y2.strip())
                        img_width = int(img_width)
                        img_height = int(img_height)

                        if self._img_dim is not None:
                            xfactor = img_width / float(self._img_dim[0])
                            yfactor = img_height / float(self._img_dim[1])
                            x1 = math.floor(float(x1) / xfactor)
                            y1 = math.floor(float(y1) / yfactor)
                            x2 = math.ceil(float(x2) / xfactor)
                            y2 = math.ceil(float(y2) / yfactor)

                        # calc width & height
                        width = x2 - x1
                        height = y2 - y1

                        # calc center
                        center_x = int(x1 + (width / 2.0))
                        center_y = int(y1 + (height / 2.0))

                        self._set_img.add(f"{set_name}/{filename}")

                        if self._labels.get(os.path.join(dirpath, filename)) == None :
                            self._labels[os.path.join(dirpath, filename)] = [[
                                ("set", set_name),
                                ("file", filename),
                                ("x1", x1),
                                ("y1", y1),
                                ("x2", x2),
                                ("y2", y2),
                                ("width", width),
                                ("height", height),
                                ("center_x", center_x),
                                ("center_y", center_y),
                                ("image_width", img_width),
                                ("image_height", img_height)]]
                        
                        else:
                            self._labels[os.path.join(dirpath, filename)] =\
                                self._labels[os.path.join(dirpath, filename)] +\
                                [[("set", set_name),
                                ("file", filename),
                                ("x1", x1),
                                ("y1", y1),
                                ("x2", x2),
                                ("y2", y2),
                                ("width", width),
                                ("height", height),
                                ("center_x", center_x),
                                ("center_y", center_y),
                                ("image_width", img_width),
                                ("image_height", img_height)]]


                print(f"read {counter} labels for set '{set_name}' from file '{filepath}'...")


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
