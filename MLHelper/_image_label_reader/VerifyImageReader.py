import os, sys
from matplotlib import pyplot as plt
from itertools import product

sys.path.append(os.environ["ROBO_AI_CODE"])
from robocup_image_reader.ImageReader import Reader



sets = ["bitbots-set00-02/", "bitbots-set00-03", "bitbots-set00-04"]
paths = [os.environ["ROBO_AI_DATA"] + iset for iset in sets]
r = Reader(paths, batch_size=8)
print(paths)

rows = 2
cols = 4

for i in range(16):
    f, axarr = plt.subplots(rows, cols)
    img_data, filepaths = r.get_next_img_batch()
    for img, (row, col) in zip(img_data, product(range(rows), range(cols))):
        axarr[row, col].imshow(img)
    plt.title("batch num {}".format(i))
    plt.show()

