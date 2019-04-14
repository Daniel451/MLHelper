import unittest
import os
import MLHelper as H
from MLHelper.datasets.bitbots import BallDatasetHandler
from pathlib import Path


class TestImageReader(unittest.TestCase):
    data = None
    test_dir = Path(os.path.realpath(__file__)).parent.absolute()
    mock_dataset_dir = test_dir.parent / "extra" / "mock-datasets"
    pathlist = [str(mock_dataset_dir)]

    @classmethod
    def setUpClass(cls):
        TestImageReader.data = H.ImgReader(TestImageReader.pathlist,
                                           batch_size=5,
                                           label_content="ball")

    def test_instantiation(self):
        self.assertIsNotNone(TestImageReader.data)

    def test_mock_size(self):
        self.assertEqual(TestImageReader.data.get_dataset_size(), 11)

    def test_mock_ball_label(self):
        dat = H.ImgReader(TestImageReader.pathlist,
                          batch_size=5,
                          label_content="ball",
                          filter_labels=True)
        self.assertEqual(dat.get_dataset_size(), 11)

    def test_batch_size(self):
        batch, paths = TestImageReader.data.get_next_img_batch()
        self.assertEqual(batch.shape[0], 5)
        batch, paths = TestImageReader.data.get_next_img_batch()
        self.assertEqual(batch.shape[0], 5)
        batch, paths = TestImageReader.data.get_next_img_batch()
        # FIXME: This fails sometimes. Race condition?
        self.assertEqual(batch.shape[0], 1)

    @unittest.skip("To be moved to interactive tests")
    def test_leipzig_ball_label(self):
        dat = H.ImgReader(BallDatasetHandler.TRAIN.LEIPZIG,
                          batch_size=8,
                          label_content="ball",
                          filter_labels=True)
        self.assertEqual(dat.get_dataset_size(), 14684)

    @unittest.skip("To be moved to interactive tests")
    def test_nagoya_and_size(self):
        dat = H.ImgReader(BallDatasetHandler.TRAIN.NAGOYA,
                          batch_size=8)
        self.assertEqual(dat.get_dataset_size(), 8243)

    @unittest.skip("To be moved to interactive tests")
    def test_montreal_and_size(self):
        dat = H.ImgReader(BallDatasetHandler.TRAIN.MONTREAL,
                          batch_size=8)
        self.assertEqual(dat.get_dataset_size(), 16866)

    @unittest.skip("To be moved to interactive tests")
    def test_challenge2018(self):
        dat = H.ImgReader(BallDatasetHandler.TRAIN.CHALLENGE_2018,
                          batch_size=8)
        dat_filtered = H.ImgReader(BallDatasetHandler.TRAIN.CHALLENGE_2018,
                                   batch_size=8,
                                   label_content="ball",
                                   filter_labels=True)
        # 35327 number of ball annotations
        # 35952 number of images total
        self.assertEqual(dat.get_dataset_size(), 35952)
        self.assertEqual(dat_filtered.get_dataset_size(), 35327)
