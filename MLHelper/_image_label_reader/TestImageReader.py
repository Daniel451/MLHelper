import os
import sys
from enum import Enum


from .ImageReader import Reader as ImgRdr


class TestDataSets(Enum):
    TestBlurReal01 = "test-blur-real-01"
    TestNagoyaGame02 = "test-nagoya-game-02"


class TestReader:

    def __init__(self):
        self._data_sets = set()


    def select_all_test_sets(self):
        self._data_sets = {m.value for m in TestDataSets}


    def add_test_set_to_selection(self, test_set: TestDataSets):
        assert isinstance(test_set, TestDataSets), f"argument 'test_set' should be of type TestDataSets (enum)," \
                                                   + f"but was '{type(test_set)}'"
        self._data_sets.add(test_set.value)


    def load_data(self, batch_size: int = 1, img_dim: tuple = (200, 150)) -> ImgRdr:
        paths = [os.path.join(os.environ["ROBO_AI_DATA"], iset) for iset in self._data_sets]

        return ImgRdr(paths, batch_size=batch_size, img_dim=img_dim, queue_size=1)
