from ._helper_functions import GAI
from ._helper_functions import PAI

try:
    import cv2
    s: str = cv2.__version__
    if not s.startswith("3"):
        raise Exception("MLHelper needs OpenCV 3 to work properly")
except Exception as e:
    print(e)
    print("please check your OpenCV installation")
