import unittest
import os
from importlib import reload
import MLHelper.datasets.bitbots._bitbots as datasets
from itertools import chain


class TestBallDatasetHandler(unittest.TestCase):
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
        mock_path = TestBallDatasetHandler.mock_path
        for collection, paths in dataset_tuple._asdict().items():
            for i, path in enumerate(paths):
                expected_prefix = mock_path + os.path.sep
                self.assertTrue(path.startswith(expected_prefix),
                                f'Expected path at {tuple_name}.{collection}[{i}] to start with "{expected_prefix}", actual: {repr(path)}')

    def _test_has_dataset_paths(self, expected_keys, dataset_tuple, tuple_name):
        for key in expected_keys:
            self.assertIn(key, dataset_tuple._fields)
            paths = getattr(dataset_tuple, key)
            self.assertIsInstance(paths, list)
            self.assertGreater(len(paths), 0, f"dataset {tuple_name}.{key} should have non-zero path items")

    def _test_dataset_all_has_all_paths(self, dataset_tuple):
        all_collections = (getattr(dataset_tuple, key) for key in dataset_tuple._fields)
        all_paths = set(path for path in chain(*all_collections))
        self.assertSetEqual(set(getattr(dataset_tuple, "ALL")), all_paths)

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
