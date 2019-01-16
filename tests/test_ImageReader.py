import unittest
import MLHelper as H
from MLHelper.datasets.bitbots import BallDatasetHandler


class TestImageReader(unittest.TestCase):
    data = None

    @classmethod
    def setUpClass(cls):
        TestImageReader.data = H.ImgReader(BallDatasetHandler.TRAIN.LEIPZIG,
                                           batch_size=8)

    def test_instantiation(self):
        self.assertIsNotNone(TestImageReader.data)

    def test_leipzig_size(self):
        self.assertEqual(TestImageReader.data.get_dataset_size(), 14684)

    def test_nagoya_and_size(self):
        dat = H.ImgReader(BallDatasetHandler.TRAIN.NAGOYA,
                          batch_size=8)
        self.assertEqual(dat.get_dataset_size(), 8243)

    def test_montreal_and_size(self):
        dat = H.ImgReader(BallDatasetHandler.TRAIN.MONTREAL,
                          batch_size=8)
        self.assertEqual(dat.get_dataset_size(), 16866)

    def test_challenge2018(self):
        dat = H.ImgReader(BallDatasetHandler.TRAIN.CHALLENGE_2018,
                          batch_size=8)
        # 35327 is the number of ball annotations
        # self.assertEqual(dat.get_dataset_size(), 35327)
        # number of images is 35952
        self.assertEqual(dat.get_dataset_size(), 35952)

    def test_batch_size(self):
        batch, paths = TestImageReader.data.get_next_img_batch()
        self.assertEqual(batch.shape[0], 8)
