import numpy as np


def batch_images_stack_plottable(img_batch: np.ndarray, split_every_n: int):
    """
    Stacks an (n,h,w,c) batch of images into one plottable 2D image.
    Images are stacked row by row, from left to right.
    If the number of images in 'img_batch' % 'split_every_n' is != 0,
    then the last j images simply get ignored, where j is the remainder
    of the modulo operation 'img_batch' % 'split_every_n'.


    :param img_batch: batch of images (n,h,w,c)
    :param split_every_n: number of images in one row
    :return: 2D image
    """
    if img_batch.shape[0] % split_every_n != 0:
        upper_bound = img_batch.shape[0] - img_batch.shape[0] % split_every_n
        img_batch = img_batch[0:upper_bound]

    stacked = np.vstack([np.hstack(e) for e in np.split(img_batch, split_every_n)])
    stacked = stacked[np.newaxis, :]

    return stacked
