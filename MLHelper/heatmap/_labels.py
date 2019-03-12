import numpy as np


def labels2D_rectangular(im_batch: np.ndarray, labels):
    # buffer
    labels2D = np.zeros(im_batch.shape[0:3], dtype=np.float32)
    shape_y = im_batch.shape[1]
    shape_x = im_batch.shape[2]
    # for k in range(0, len(labels)):
    #     for j in range (0, len(labels[k])):
    #         labels[k] = dict(labels[k])
    # iterate over all batches
    for i, d in enumerate(labels):
        print("i:" + str(i) + " d:" + str(d))
        for k in range(0, len(labels)):
            print("k:" + str(k))
            for j in range (0, len(labels[k])):
                print("j:" + str(j))
                labels[j] = dict(labels[j])
                print("d:" + str(d) + " i:" + str(i) + " k:" + str(k) + " j:" + str(j))
                labels2D[i, d["y1"]: d["y1"] + d["height"], d["x1"]: d["x1"] + d["width"]] = 1.0

    return labels2D.reshape(-1, shape_y, shape_x, 1)


def labels2D_circular(im_batch, labels):
    # buffer
    labels2D = np.zeros(im_batch.shape[0:3], dtype=np.float32)
    shape_y = im_batch.shape[1]
    shape_x = im_batch.shape[2]

    for i, d in enumerate(labels):
        y = np.arange(0, shape_y)
        x = np.arange(0, shape_x)

        cy = d["center_y"]
        cx = d["center_x"]
        r = int(((d["width"] / 2) + (d["height"] / 2)) / 2) + 1

        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
        labels2D[i][mask] = 1.0

    return labels2D.reshape(-1, shape_y, shape_x, 1)
