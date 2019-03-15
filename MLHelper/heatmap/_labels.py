import numpy as np


def labels2D_rectangular(im_batch: np.ndarray, labels):
    # buffer
	labels2D = np.zeros(im_batch.shape[0:3], dtype=np.float32)
	shape_y = im_batch.shape[1]
	shape_x = im_batch.shape[2]
	for i, d in enumerate(labels):
		for j in range (0, len(d)):
			b = dict(d[j])
			labels2D[i, b["y1"]: b["y1"] + b["height"], b["x1"]: b["x1"] + b["width"]] = 1.0

	return labels2D.reshape(-1, shape_y, shape_x, 1)


def labels2D_circular(im_batch, labels):
    # buffer
    labels2D = np.zeros(im_batch.shape[0:3], dtype=np.float32)
    shape_y = im_batch.shape[1]
    shape_x = im_batch.shape[2]

    for i, d in enumerate(labels):
        y = np.arange(0, shape_y)
        x = np.arange(0, shape_x)
        for j in range (0, len(d)):
            b = dict(d[j])

            cy = b["center_y"]
            cx = b["center_x"]
            r = int(((b["width"] / 2) + (b["height"] / 2)) / 2) + 1

            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
            labels2D[i][mask] = 1.0

    return labels2D.reshape(-1, shape_y, shape_x, 1)
