from ._helper_functions import GAI
from ._helper_functions import PAI
from ._image_label_reader.ImageLabelReader import DataObject as ImgLblReader
from ._image_label_reader.ImageReader import Reader as ImgReader

# module structure:
# * Python 3.3+ treats all folders as packages, thus empty __init__.py files are no longer necessary


# check for OpenCV 3
try:
    import cv2 as _cv2
    _s: str = _cv2.__version__
    if not _s.startswith("3"):
        raise Exception("MLHelper needs OpenCV 3 to work properly")
    del _cv2
    del _s
except Exception as e:
    print(e)
    print("please check your OpenCV installation")


