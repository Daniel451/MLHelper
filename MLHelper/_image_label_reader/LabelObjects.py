class LabelBoundingBox2D:

    def __init__(self, x1: int, y1: int, x2: int, y2: int,
                 image_width: int = None, image_height: int = None,
                 set_name: str = None, filename: str = None):
        # label coordinates
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        # other label attributes
        self.width = x2 - x1
        self.height = y2 - y1
        self.center_x = int(x1 + (self.width / 2.0))
        self.center_y = int(y1 + (self.height / 2.0))

        # original image attributes
        self.image_width = image_width
        self.image_height = image_height

        # set and file information
        self.name_set = set_name
        self.name_file = filename


    def __eq__(self, other):
        if not isinstance(other, LabelBoundingBox2D):
            return False

        return self.x1 == other.x1 \
               and self.y1 == other.y1 \
               and self.x2 == other.x2 \
               and self.y2 == other.y2
