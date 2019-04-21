import unittest
import os
import numpy as np
import MLHelper as H
from itertools import product
from MLHelper.datasets.bitbots import BallDatasetHandler
from pathlib import Path


class TestImageReader(unittest.TestCase):
    data = None
    test_dir = Path(os.path.realpath(__file__)).parent.absolute()
    mock_dataset_dir = test_dir.parent / "extra" / "mock-datasets"
    pathlist = [str(mock_dataset_dir)]
    batch_size = 5

    @classmethod
    def setUpClass(cls):
        TestImageReader.data = H.ImgReader(TestImageReader.pathlist,
                                           batch_size=TestImageReader.batch_size,
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
        """
        run get_next_batch several times
        -> it should always return *extactly* batch_size samples
        """
        for i in range(20):
            batch, paths = TestImageReader.data.get_next_img_batch()
            self.assertEqual(batch.shape[0], TestImageReader.batch_size, msg=f"iteration {i}")

    def test_random_order(self):
        path_buffer = list()
        for i in range(10):
            _, paths = TestImageReader.data.get_next_img_batch()
            path_buffer.append(paths)

        for a, b in product(path_buffer, path_buffer):
            if id(a) == id(b):
                continue

            self.assertTrue(np.any(a != b), msg="this tests the random ordering of paths for each"
                                                +" batch. This might fail with a very low probability.")

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
