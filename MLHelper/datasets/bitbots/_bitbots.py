import os
from collections import namedtuple
from itertools import chain


def prepend_data_path(paths):
    return (os.path.join(os.environ["ROBO_AI_DATA"], path) for path in paths)


def collection_to_id(collection_name):
    return _collection_to_id[collection_name]


def collections_to_paths(collection_names):
    return frozenset(prepend_data_path(str(collection_to_id(collection_name)) for collection_name in collection_names))


_collection_to_id = {
    "bitbots-set00-01": 5,
    "bitbots-set00-02": 6,
    "bitbots-set00-03": 7,
    "bitbots-set00-04": 12,
    "bitbots-set00-05": 13,
    "bitbots-set00-06": 14,
    "bitbots-set00-07": 15,
    "bitbots-set00-08": 16,
    "bitbots-set00-09": 17,
    "bitbots-set00-10": 18,
    "bitbots-set00-11": 29,
    "bitbots-set00-12": 30,
    "bitbots-set00-13": 31,
    "bitbots-set00-14": 32,
    "bitbots-set00-15": 33,
    "bitbots-nagoya-euro-ball-game-01": 36,
    "bitbots-nagoya-sequences-euro-ball-robot-1": 153,
    "bitbots-nagoya-sequences-jasper-euro-ball-1": 157,
    "bitbots-nagoya-sequences-jasper-kicking-euro-ball": 156,
    "bitbots-nagoya-sequences-niklas": 489,
    "bitbots-nagoya-sequences-misc-ball-1": 154,
    "bitbots-nagoya-sequences-misc-ball-robot-1": 155,
    "bitbots-nagoya-euro-ball-02": 490,
    "bitbots-2018-iran-01": 160,
    "bitbots-2018-iran-02": 161,
    "bitbots-2018-iran-03": 162,
    "bitbots-2018-iran-04": 163,
    "bitbots-2018-iran-05": 164,
    "bitbots-2018-iran-06": 165,
    "bitbots-2018-iran-game-01-mrl-bitbots": 166,
    "bitbots-fifa18-high-res-02": 259,
    "bitbots-montreal-game01": 260,
    "bitbots-montreal-game02": 261,
    "bitbots-montreal-game03": 474,
    "bitbots-montreal-game04": 262,
    "bitbots-montreal-game05": 475,
    "bitbots-montreal-game06": 476,
    "bitbots-montreal-game07": 477,
    "bitbots-montreal-game08": 478,
    "bitbots-montreal-game09": 479,
    "bitbots-montreal-game10": 491,
    "bitbots-fifa18-high-res-01": 196,
    "bitbots-fifa18-davros-01": 197,
    "test-nagoya-game-02": 81,
    "test-wolves-01": 159,
    "bitbots-2018-iran-20-minibot": 191,
    "bitbots-lab-euro-ball-robot-concealed": 182,
    "noised-test-nagoya-game-02": 499,
    "noised-test-wolves-01": 500,
    "test-blur-real-01": 501,
    "leipzig-ball-negative-01": 502,
    "bitbots-nagoya-ball-negative-01": 488,
    "bitbots-lab-ball-negative-cam-01": 503,
    "bitbots-nagoya-random-01": 504,
    "bitbots-nagoya-random-02": 505,
    "bitbots-nagoya-random-03": 506,
    "bitbots-nagoya-random-04": 507,
    "bitbots-nagoya-random-05": 508,
    "bitbots-nagoya-random-06": 509,
    "bitbots-nagoya-random-07": 510,
    "bitbots-lab-ball-random-01": 511,
}


#############
### TRAIN ###
#############

_challenge2018 = frozenset(set.union(
    set(["bitbots-set00-01", "bitbots-set00-02", "bitbots-set00-03", "bitbots-set00-04",
         "bitbots-set00-05", "bitbots-set00-06", "bitbots-set00-07", "bitbots-set00-08",
         "bitbots-set00-09", "bitbots-set00-10", "bitbots-set00-11", "bitbots-set00-12",
         "bitbots-set00-13", "bitbots-set00-14", "bitbots-set00-15"]),
    set(["bitbots-nagoya-euro-ball-game-01",
         "bitbots-nagoya-sequences-euro-ball-robot-1",
         "bitbots-nagoya-sequences-jasper-euro-ball-1",
         "bitbots-nagoya-sequences-jasper-kicking-euro-ball",
         "bitbots-nagoya-sequences-misc-ball-1",
         "bitbots-nagoya-sequences-misc-ball-robot-1"]),
    set(["bitbots-2018-iran-01",
         "bitbots-2018-iran-03",
         "bitbots-2018-iran-04",
         "bitbots-2018-iran-05",
         "bitbots-2018-iran-06",
         "bitbots-2018-iran-game-01-mrl-bitbots"]),
    set(["bitbots-fifa18-high-res-01",
         "bitbots-fifa18-davros-01"])
))

_train_data_leipzig = frozenset(
    ["bitbots-set00-01", "bitbots-set00-02", "bitbots-set00-03", "bitbots-set00-04",
     "bitbots-set00-05", "bitbots-set00-06", "bitbots-set00-07", "bitbots-set00-08",
     "bitbots-set00-09", "bitbots-set00-10", "bitbots-set00-11", "bitbots-set00-12",
     "bitbots-set00-13", "bitbots-set00-14", "bitbots-set00-15"]
)

_train_data_nagoya = frozenset(
    ["bitbots-nagoya-euro-ball-game-01",
     "bitbots-nagoya-sequences-euro-ball-robot-1",
     "bitbots-nagoya-sequences-jasper-euro-ball-1",
     "bitbots-nagoya-sequences-jasper-kicking-euro-ball",
     "bitbots-nagoya-sequences-niklas",
     "bitbots-nagoya-sequences-misc-ball-1",
     "bitbots-nagoya-sequences-misc-ball-robot-1",
     "bitbots-nagoya-euro-ball-02"]
)

_train_data_montreal = frozenset(
    ["bitbots-fifa18-high-res-02",
     "bitbots-montreal-game01",
     "bitbots-montreal-game02",
     "bitbots-montreal-game03",
     "bitbots-montreal-game04",
     "bitbots-montreal-game05",
     "bitbots-montreal-game06",
     "bitbots-montreal-game07",
     "bitbots-montreal-game08",
     "bitbots-montreal-game09",
     "bitbots-montreal-game10"]
)

_train_data_iran = frozenset(
    ["bitbots-2018-iran-01",
     "bitbots-2018-iran-02",
     "bitbots-2018-iran-03",
     "bitbots-2018-iran-04",
     "bitbots-2018-iran-05",
     "bitbots-2018-iran-06",
     "bitbots-2018-iran-game-01-mrl-bitbots"]
)

_train_data_bitbotslab = frozenset(
    ["bitbots-fifa18-high-res-01",
     "bitbots-fifa18-davros-01"]
)

############
### TEST ###
############

_test_nagoya_game_02 = frozenset(["test-nagoya-game-02"])
_test_wolves_01 = frozenset(["test-wolves-01"])
_test_2018_iran_minibot_20 = frozenset(["bitbots-2018-iran-20-minibot"])

_test_bitbotslab_robot_concealed = frozenset(["bitbots-lab-euro-ball-robot-concealed"])

###################
### TEST NOISED ###
###################

_noised_test_nagoya_game_02 = frozenset(["noised-test-nagoya-game-02"])
_noised_test_wolves_01 = frozenset(["noised-test-wolves-01"])

_noised_test_real_01 = frozenset(["test-blur-real-01"])

################
### NEGATIVE ###
################

_negative_data_leipzig = frozenset(["leipzig-ball-negative-01"])

_negative_data_nagoya = frozenset(["bitbots-nagoya-ball-negative-01"])

_negative_data_bitbotslab = frozenset(["bitbots-lab-ball-negative-cam-01"])

##############
### RANDOM ###
##############

_random_data_nagoya = frozenset(
    ["bitbots-nagoya-random-01",
     "bitbots-nagoya-random-02",
     "bitbots-nagoya-random-03",
     "bitbots-nagoya-random-04",
     "bitbots-nagoya-random-05",
     "bitbots-nagoya-random-06",
     "bitbots-nagoya-random-07"]
)

_random_data_bitbotslab = frozenset(["bitbots-lab-ball-random-01"])


class BallDatasetHandler:
    _train_tuple = namedtuple("TRAIN", ["LEIPZIG", "NAGOYA", "IRAN", "MONTREAL", "BITBOTSLAB",
                                        "CHALLENGE_2018",
                                        "ALL"])

    TRAIN = _train_tuple(
        LEIPZIG=collections_to_paths(_train_data_leipzig),
        NAGOYA=collections_to_paths(_train_data_nagoya),
        IRAN=collections_to_paths(_train_data_iran),
        MONTREAL=collections_to_paths(_train_data_montreal),
        BITBOTSLAB=collections_to_paths(_train_data_bitbotslab),
        CHALLENGE_2018=collections_to_paths(_challenge2018),
        ALL=collections_to_paths(frozenset.union(_train_data_leipzig, _train_data_nagoya, _train_data_iran,
                                        _train_data_bitbotslab, _train_data_montreal))
    )

    _test_tuple = namedtuple("TEST", ["NAGOYA", "WOLVES", "IRAN", "BITBOTSLAB_CONCEALED",
                                      "CHALLENGE_2018",
                                      "ALL"])

    TEST = _test_tuple(
        NAGOYA=collections_to_paths(_test_nagoya_game_02),
        WOLVES=collections_to_paths(_test_wolves_01),
        IRAN=collections_to_paths(_test_2018_iran_minibot_20),
        BITBOTSLAB_CONCEALED=collections_to_paths(_test_bitbotslab_robot_concealed),
        CHALLENGE_2018=collections_to_paths(frozenset.union(_test_nagoya_game_02, _test_wolves_01)),
        ALL=collections_to_paths(frozenset.union(_test_nagoya_game_02, _test_wolves_01, _test_2018_iran_minibot_20,
                                           _test_bitbotslab_robot_concealed))
    )

    _test_noised_tuple = namedtuple("TEST_NOISED", ["NAGOYA", "WOLVES", "REAL",
                                                    "CHALLENGE_2018",
                                                    "ALL"])

    TEST_NOISED = _test_noised_tuple(
        NAGOYA=collections_to_paths(_noised_test_nagoya_game_02),
        WOLVES=collections_to_paths(_noised_test_wolves_01),
        REAL=collections_to_paths(_noised_test_real_01),
        CHALLENGE_2018=collections_to_paths(frozenset.union(_noised_test_nagoya_game_02, _noised_test_wolves_01)),
        ALL=collections_to_paths(frozenset.union(_noised_test_nagoya_game_02, _noised_test_wolves_01, _noised_test_real_01))
    )


class NegativeBallDatasetHandler:
    _data_tuple = namedtuple("DATATUPLE", ["LEIPZIG", "NAGOYA", "BITBOTSLAB", "ALL"])

    DATA = _data_tuple(LEIPZIG=collections_to_paths(_negative_data_leipzig),
                       NAGOYA=collections_to_paths(_negative_data_nagoya),
                       BITBOTSLAB=collections_to_paths(_negative_data_bitbotslab),
                       ALL=collections_to_paths(frozenset.union(_negative_data_leipzig, _negative_data_nagoya, _negative_data_bitbotslab))
    )


class RandomDatasetHandler:
    _data_tuple = namedtuple("DATATUPLE", ["NAGOYA", "BITBOTSLAB", "ALL"])

    DATA = _data_tuple(NAGOYA=collections_to_paths(_random_data_nagoya),
                       BITBOTSLAB=collections_to_paths(_random_data_bitbotslab),
                       ALL=collections_to_paths(frozenset.union(_random_data_nagoya, _random_data_bitbotslab))
    )
