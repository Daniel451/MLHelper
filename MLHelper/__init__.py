# top-level helper
from ._helper_functions import GAI as GAI
from ._helper_functions import PAI as PAI

# top-level objects and functions
from ._image_label_reader.ImageLabelReader import DataObject as ImgLblReader
from ._image_label_reader.ImageReader import Reader as ImgReader
from ._image_label_reader.LabelReader import Reader as LblReader
from ._image_label_reader.ImageBatch import ImageBatch
from ._image_label_reader.LabelObjects import LabelBoundingBox2D

# submodules
from . import heatmap
from . import noise
from . import dir
from . import img


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
    exit("please check your OpenCV installation")



