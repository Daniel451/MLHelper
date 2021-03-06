import unittest
import os
from importlib import reload
import MLHelper.datasets.bitbots._bitbots as datasets
from itertools import chain


class TestDatasetHandler(unittest.TestCase):
    mock_path = 'mock_path'
    orig_path = None

    @classmethod
    def setUpClass(cls):
        cls.orig_path = os.environ['ROBO_AI_DATA']
        os.environ['ROBO_AI_DATA'] = cls.mock_path
        reload(datasets)

    @classmethod
    def tearDownClass(cls):
        os.environ['ROBO_AI_DATA'] = cls.orig_path
        reload(datasets)

    def _test_paths_prepended(self, dataset_tuple, tuple_name):
        mock_path = TestDatasetHandler.mock_path
        for collection_name, collection in dataset_tuple._asdict().items():
            for i, path in enumerate(collection.to_paths()):
                expected_prefix = mock_path + os.path.sep
                self.assertTrue(path.startswith(expected_prefix),
                                f'Expected path at {tuple_name}.{collection_name}[{i}] to start with "{expected_prefix}", actual: {repr(path)}')

    def _test_has_dataset_paths(self, expected_keys, dataset_tuple, tuple_name):
        for key in expected_keys:
            self.assertIn(key, dataset_tuple._fields)
            collection = getattr(dataset_tuple, key)
            self.assertIsInstance(collection, datasets.ImagesetCollection)
            paths = collection.to_paths()
            self.assertIsInstance(paths, list)
            self.assertGreater(len(paths), 0, f"dataset {tuple_name}.{key} should have non-zero path items")

    def _test_dataset_all_has_all_paths(self, dataset_tuple):
        all_collections = (getattr(dataset_tuple, key) for key in dataset_tuple._fields)
        all_collection_paths = (collection.to_paths() for collection in all_collections)
        all_paths = set(path for path in chain(*all_collection_paths))
        self.assertSetEqual(set(getattr(dataset_tuple, "ALL").to_paths()), all_paths)


class TestBallDatasetHandler(TestDatasetHandler):
    def test_train_paths_prepended(self):
        self._test_paths_prepended(datasets.BallDatasetHandler.TRAIN, 'TRAIN')

    def test_test_paths_prepended(self):
        self._test_paths_prepended(datasets.BallDatasetHandler.TEST, 'TEST')

    def test_test_noised_paths_prepended(self):
        self._test_paths_prepended(datasets.BallDatasetHandler.TEST_NOISED, 'TEST_NOISED')

    def test_has_expected_train_datasets(self):
        expected_keys = ("LEIPZIG", "NAGOYA", "IRAN", "MONTREAL", "BITBOTSLAB",
                         "CHALLENGE_2018", "ALL")
        dataset_tuple = datasets.BallDatasetHandler.TRAIN
        self._test_has_dataset_paths(expected_keys, datasets.BallDatasetHandler.TRAIN, 'TRAIN')
        self._test_dataset_all_has_all_paths(dataset_tuple)

    def test_has_expected_test_datasets(self):
        expected_keys = ("NAGOYA", "WOLVES", "IRAN", "BITBOTSLAB_CONCEALED",
                         "CHALLENGE_2018", "ALL")
        dataset_tuple = datasets.BallDatasetHandler.TEST
        self._test_has_dataset_paths(expected_keys, dataset_tuple, 'TEST')
        self._test_dataset_all_has_all_paths(dataset_tuple)

    def test_has_expected_test_noised_datasets(self):
        expected_keys = ("NAGOYA", "WOLVES", "REAL",
                         "CHALLENGE_2018", "ALL")
        dataset_tuple = datasets.BallDatasetHandler.TEST_NOISED
        self._test_has_dataset_paths(expected_keys, dataset_tuple, 'TEST_NOISED')
        self._test_dataset_all_has_all_paths(dataset_tuple)


class TestNegativeBallDatasetHandler(TestDatasetHandler):
    def test_data_paths_prepended(self):
        self._test_paths_prepended(datasets.NegativeBallDatasetHandler.DATA, 'DATA')

    def test_has_expected_data_datasets(self):
        expected_keys = ("LEIPZIG", "NAGOYA", "BITBOTSLAB", "ALL")
        dataset_tuple = datasets.BallDatasetHandler.TRAIN
        self._test_has_dataset_paths(expected_keys, datasets.NegativeBallDatasetHandler.DATA, 'DATA')
        self._test_dataset_all_has_all_paths(dataset_tuple)


class TestRandomDatasetHandler(TestDatasetHandler):
    def test_data_paths_prepended(self):
        self._test_paths_prepended(datasets.RandomDatasetHandler.DATA, 'DATA')

    def test_has_expected_data_datasets(self):
        expected_keys = ("NAGOYA", "BITBOTSLAB", "ALL")
        dataset_tuple = datasets.BallDatasetHandler.TRAIN
        self._test_has_dataset_paths(expected_keys, datasets.RandomDatasetHandler.DATA, 'DATA')
        self._test_dataset_all_has_all_paths(dataset_tuple)
