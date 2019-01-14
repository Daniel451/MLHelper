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

    def test_batch_size(self):
        batch, paths = TestImageReader.data.get_next_img_batch()
        self.assertEqual(batch.shape[0], 8)
