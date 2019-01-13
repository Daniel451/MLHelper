import numpy as np


def PAI(arr: np.ndarray, name: str = "") -> None:
    """
    prints array information
    """
    print(GAI(arr, name))


def GAI(arr: np.ndarray, name: str = "") -> str:
    """
    returns formatted array information string
    """
    shapestr = str(arr.shape).rjust(20, " ")
    dtypestr = str(arr.dtype).rjust(10, " ")

    pstr = ""
    if name != "":
        pstr = "[{}] ".format(name).ljust(20, " ")
    pstr += "shape: {} | dtype: {} | min: {:>10.5f} | max: {:>10.5f} | mean: {:>10.5f} | sum: {:>10.2f}" \
        .format(shapestr, dtypestr, np.min(arr), np.max(arr), np.mean(arr), np.sum(arr))

    return pstr
