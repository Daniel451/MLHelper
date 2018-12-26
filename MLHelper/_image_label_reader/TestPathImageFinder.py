import unittest
import os

from .PathImageFinder import ImgFind



class TestPathImageFinder(unittest.TestCase):

    def setUp(self):
        self._path = os.environ["ROBO_AI_DATA"] + "bitbots-set00-02/"
        isets = ["bitbots-set00-02", "bitbots-set00-03/", "bitbots-set00-04"]
        self._pathlist = [os.environ["ROBO_AI_DATA"] + imgpath for imgpath in isets]

    def test_all_png_filenames(self):
        # get the contents of the directory
        contents = os.listdir(self._path)

        # get all png files
        contents = list(filter(lambda x: x.lower().endswith(".png"), contents))

        # check consistency of image set
        # only works if all images have labels
        for f in contents:
            self.assertTrue(self._path + f in ImgFind.find_pngs([self._path]))

    def test_path_validity(self):
        self.assertEqual([self._path], ImgFind.check_path_validity([self._path]))


