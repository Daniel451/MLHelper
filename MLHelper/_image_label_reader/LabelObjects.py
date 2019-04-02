class LabelBoundingBox2D:

    def __init__(self, x1: int, y1: int, x2: int, y2: int,
                 image_width: int = None, image_height: int = None,
                 set_name: str = None, filename: str = None):
        # label coordinates
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

        # other label attributes
        self._width = x2 - x1
        self._height = y2 - y1
        self._center_x = int(x1 + (self.width / 2.0))
        self._center_y = int(y1 + (self.height / 2.0))

        # original image attributes
        self._image_width = image_width
        self._image_height = image_height

        # set and file information
        self._name_set = set_name
        self._name_file = filename


    @property
    def x1(self):
        return self._x1


    @property
    def y1(self):
        return self.y1


    @property
    def x2(self):
        return self.x2


    @property
    def y2(self):
        return self.y2


    @property
    def width(self):
        return self._width


    @property
    def height(self):
        return self._height


    @property
    def center_x(self):
        return self._center_x


    @property
    def center_y(self):
        return self._center_y


    @property
    def image_width(self):
        return self._image_width


    @property
    def image_height(self):
        return self._image_height


    @property
    def name_set(self):
        return self._name_set


    @property
    def name_file(self):
        return self._name_file


    def __eq__(self, other):
        if not isinstance(other, LabelBoundingBox2D):
            return False

        return self.x1 == other.x1 \
               and self.y1 == other.y1 \
               and self.x2 == other.x2 \
               and self.y2 == other.y2
