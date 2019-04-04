from tqdm import tqdm
import numpy as np
import glob
import argparse
import os
import sys
import cv2
from shutil import copyfile
import MLHelper as H



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageset", type=str, required=True, help="path to imageset")
    parser.add_argument("--save_path", type=str, required=True, help="path where the noised copy should be stored")
    args = parser.parse_args()

    if not os.path.exists(args.imageset):
        raise IOError(f"path '{args.imageset}' does not exist.")

    if os.path.exists(args.save_path):
        H.dir.clean_directory(args.save_path)
    else:
        os.makedirs(args.save_path)

    # gather png files in imageset path
    ipath = os.path.join(args.imageset, "*.png")
    imgs = glob.glob(ipath)

    # gather label files
    labels = glob.glob(os.path.join(args.imageset, "*.txt"))

    # check path for files
    if not len(imgs) > 0:
        raise Exception(f"no png files found for path '{args.imageset}'")


    for filepath in tqdm(imgs):
        filename = os.path.basename(filepath)
        im = cv2.imread(filepath)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32) / 255.0

        noised = H.noise.random_motionblur_img(im)

        noised = noised * 255.0
        noised = noised.astype(np.uint8)
        noised = cv2.cvtColor(noised, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(args.save_path, filename), noised)

    # copy labels
    for filepath in labels:
        filename = os.path.basename(filepath)
        copyfile(filepath, os.path.join(args.save_path, filename))

    print()
    print("done.")






