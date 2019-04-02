from collections import namedtuple


LabelDataTuple = namedtuple("LabelDataTuple",
                            [
                                "set",
                                "file",
                                "x1",
                                "y1",
                                "x2",
                                "y2",
                                "width",
                                "height",
                                "center_x",
                                "center_y",
                                "image_width",
                                "image_height"
                            ])
