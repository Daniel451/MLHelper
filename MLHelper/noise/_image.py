import numpy as np
import random
import cv2


def overwrite_black_areas_from_im1_with_im2(im1, im2):
    """
    overwrites black (=0) areas in image1 with information from image2
    :param im1: image /np.ndarray
    :param im2: image /np.ndarray
    :return:
    """
    idx = np.where(im1 == 0)
    im1[idx] = im2[idx]

    return im1


def motionblur_img(im: np.ndarray, tr_x: float, tr_y: float, k=None):
    """
    motion blur's an image using image translating, blending and gaussian blurring

    :param im: input image
    :param tr_x: translation on x-axis
    :param tr_y: translation on y-axis
    :param k: gaussian noise kernel (optional; if None a random kernel will be computed)
    :return: blurred image
    """

    # extract height and width
    h, w = im.shape[0:2]

    # translation kernels
    k1 = np.eye(2, 3, dtype=np.float32)
    k1[0, 2] = tr_x
    k1[1, 2] = tr_y
    k2 = np.eye(2, 3, dtype=np.float32)
    k2[0, 2] = -tr_x
    k2[1, 2] = -tr_y

    # translation1
    tr1 = cv2.warpAffine(im, k1, (w, h))
    tr2 = cv2.warpAffine(im, k2, (w, h))

    # fix black borders of translated images
    tr1 = overwrite_black_areas_from_im1_with_im2(tr1, im)
    tr2 = overwrite_black_areas_from_im1_with_im2(tr2, im)

    # blend translated images into each other
    tr_blended = cv2.addWeighted(tr1, 0.5, tr2, 0.5, 0.0)

    # blend original image with translations
    blended = cv2.addWeighted(im, 0.5, tr_blended, 0.5, 0)

    # apply gaussian noise
    if k is None:
        ksize = random.choice(range(3, 32, 2))
        sigma = np.random.uniform(1.0, ksize, 1)
        k = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
    # noised = cv2.GaussianBlur(blended, (ksize, ksize), random.randint(3, 20))
    noised = cv2.filter2D(blended, -1, k)

    return noised


def random_motionblur_img(im: np.ndarray):
    """
    Randomly motion blurs an image.

    :param im: input image
    :return: motion blurred image
    """
    # draw random number on a circle
    t = np.random.random() * 2 * np.pi
    v_x = np.cos(t)
    v_y = np.sin(t)

    img_size = (np.max(im.shape[0:2]) - np.min(im.shape[0:2])) / 2 + np.min(im.shape[0:2])

    scale = random.randint(1, int(img_size / 25.0))

    return motionblur_img(im, v_x * scale, v_y * scale)


def random_motionblur_images(imgs: np.ndarray, dtype: np.dtype = np.float32):
    """
    Randomly motion blurs multiple images.

    :param imgs: input images
    :param dtype: dtype of output images
    :return: motion blurred images
    """
    buffer = list()

    for im in imgs:
        buffer.append(random_motionblur_img(im))

    arr = np.array(buffer).astype(dtype)

    return arr
