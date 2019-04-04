import unittest
import os
import time
import MLHelper as H
from MLHelper.datasets.bitbots import BallDatasetHandler


class TestLabelReader(unittest.TestCase):
    data = None


    @classmethod
    def setUpClass(cls):
        TestLabelReader.data = H.LblReader(BallDatasetHandler.TRAIN.LEIPZIG,
                                           img_dim=(800, 600),
                                           label_content="ball")


    def test_instantiation(self):
        self.assertIsNotNone(TestLabelReader.data)


    def test_leipzig_size(self):
        self.assertEqual(len(TestLabelReader.data.get_set_img()), 14684)


    def test_multiple_goalpost(self):
        dat = H.LblReader([os.path.join(os.environ["ROBO_AI_DATA"], "bitbots-set00-01")],
                          img_dim=(800, 600),
                          label_content="goalpost")

        imgs = ["jan16_seq__000.531.png"]
        paths = [os.path.join(os.environ["ROBO_AI_DATA"], "bitbots-set00-01", imgs[0])]

        labels = dat.get_labels_for_batch(paths)

        # image 'jan16_seq__000.531.png' contains two goalposts - check for two labels
        self.assertEqual(len(labels[0]), 2)

        # extract label objects
        # left goal post: x1=26 and y1=0
        if abs(labels[0][0].x1 - 26) <= 2 and abs(labels[0][0].y1 - 0) <= 2:
            # if x1 and y1 of labels[0][0] matched, then labels[0][0] is the left goalpost
            goalpost1 = labels[0][0]
            goalpost2 = labels[0][1]
        else:
            # if it did not match, then just assume right goalpost is labels[0][0]
            goalpost2 = labels[0][0]
            goalpost1 = labels[0][1]

        # check left goalpost information
        self.assertAlmostEqual(goalpost1.x1, 26, delta=2)
        self.assertAlmostEqual(goalpost1.y1, 0, delta=2)
        self.assertAlmostEqual(goalpost1.x2, 110, delta=2)
        self.assertAlmostEqual(goalpost1.y2, 150, delta=2)

        self.assertAlmostEqual(goalpost1.center_x, 68, delta=2)
        self.assertAlmostEqual(goalpost1.center_y, 75, delta=2)
        self.assertAlmostEqual(goalpost1.width, 84, delta=2)
        self.assertAlmostEqual(goalpost1.height, 150, delta=2)

        # check right goalpost information
        self.assertAlmostEqual(goalpost2.x1, 731, delta=2)
        self.assertAlmostEqual(goalpost2.y1, 0, delta=2)
        self.assertAlmostEqual(goalpost2.x2, 799, delta=2)
        self.assertAlmostEqual(goalpost2.y2, 123, delta=2)

        self.assertAlmostEqual(goalpost2.center_x, 765, delta=2)
        self.assertAlmostEqual(goalpost2.center_y, 61.5, delta=2)
        self.assertAlmostEqual(goalpost2.width, 68, delta=2)
        self.assertAlmostEqual(goalpost2.height, 123, delta=2)


    def test_leipzig_labels_for_two_images(self):
        imgs = ["img_fr_nov15_000.000.png", "img_fr_nov15_000.001.png"]
        paths = [os.path.join(os.environ["ROBO_AI_DATA"], "bitbots-set00-01", i) for i in imgs]

        labels = TestLabelReader.data.get_labels_for_batch(paths)

        # check for exactly two labels
        self.assertEqual(len(labels), 2)

        # extract label objects
        label_img1 = labels[0][0]
        label_img2 = labels[1][0]

        # image1
        # check information
        self.assertEqual(label_img1.name_set, "bitbots-set00-01")
        self.assertEqual(label_img1.name_file, imgs[0])
        # check coordinates
        self.assertAlmostEqual(label_img1.x1, 338, delta=1)
        self.assertAlmostEqual(label_img1.y1, 163, delta=1)
        self.assertAlmostEqual(label_img1.x2, 443, delta=1)
        self.assertAlmostEqual(label_img1.y2, 277, delta=1)
        # check image properties
        self.assertEqual(label_img1.image_width, 800)
        self.assertEqual(label_img1.image_height, 600)
        # check bounding box properties
        self.assertAlmostEqual(label_img1.center_x, 390.5, delta=1)
        self.assertAlmostEqual(label_img1.center_y, 220, delta=1)
        self.assertAlmostEqual(label_img1.width, 105, delta=1)
        self.assertAlmostEqual(label_img1.height, 114, delta=1)

        # image2
        # check information
        self.assertEqual(label_img2.name_set, "bitbots-set00-01")
        self.assertEqual(label_img2.name_file, imgs[1])
        # check coordinates
        self.assertAlmostEqual(label_img2.x1, 498, delta=1)
        self.assertAlmostEqual(label_img2.y1, 152, delta=1)
        self.assertAlmostEqual(label_img2.x2, 609, delta=1)
        self.assertAlmostEqual(label_img2.y2, 262, delta=1)
        # check image properties
        self.assertEqual(label_img2.image_width, 800)
        self.assertEqual(label_img2.image_height, 600)
        # check bounding box properties
        self.assertAlmostEqual(label_img2.center_x, 553.5, delta=1)
        self.assertAlmostEqual(label_img2.center_y, 207, delta=1)
        self.assertAlmostEqual(label_img2.width, 111, delta=1)
        self.assertAlmostEqual(label_img2.height, 110, delta=1)

