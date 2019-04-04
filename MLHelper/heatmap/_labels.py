import numpy as np
from typing import List
from .._image_label_reader.LabelObjects import LabelBoundingBox2D


def labels2D_rectangular(im_batch: np.ndarray, labels: List[List[LabelBoundingBox2D]]):
    # buffer
    labels2D = np.zeros(im_batch.shape[0:3], dtype=np.float32)
    shape_y = im_batch.shape[1]
    shape_x = im_batch.shape[2]

    # iterate over all images in batch
    for i, label_list in enumerate(labels):
        # iterate over all LabelDataTuples for current image
        for bbx in label_list:
            labels2D[i, bbx.y1 : bbx.y1+bbx.height+1, bbx.x1 : bbx.x1+bbx.width+1] = 1.0

    return labels2D.reshape(-1, shape_y, shape_x, 1)


def labels2D_circular(im_batch, labels: List[List[LabelBoundingBox2D]]):
    # buffer
    labels2D = np.zeros(im_batch.shape[0:3], dtype=np.float32)
    shape_y = im_batch.shape[1]
    shape_x = im_batch.shape[2]

    for i, label_list in enumerate(labels):
        y = np.arange(0, shape_y)
        x = np.arange(0, shape_x)
        for bbx in label_list:
            cy = bbx.center_y
            cx = bbx.center_x
            r = int(((bbx.width / 2) + (bbx.height / 2)) / 2) + 1

            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
            labels2D[i][mask] = 1.0

    return labels2D.reshape(-1, shape_y, shape_x, 1)
