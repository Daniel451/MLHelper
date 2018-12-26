import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from itertools import product

from .ImageLabelReader import DataObject

batch_size = 12
imageset = ["bitbots-set00-01", "bitbots-set00-02", "bitbots-set00-03", "bitbots-set00-04", "bitbots-set00-05",
            "bitbots-set00-06", "bitbots-set00-07", "bitbots-set00-08", "bitbots-set00-09", "bitbots-set00-10",
            "bitbots-set00-11", "bitbots-set00-12", "bitbots-set00-13", "bitbots-set00-14", "bitbots-set00-15"]
# imageset = ["test-nagoya-game-02"]
pathlist = [os.path.join(os.environ["ROBO_AI_DATA"], iset) for iset in imageset]
d = DataObject(pathlist, batch_size, 8)

def plot_batch():
    batch = d.get_next_batch()
    # Create figure and axes
    for i, (row, col), labels in zip(range(batch_size), product(range(3), range(4)), batch.get_labels()):
        # Display the image
        axarr[row, col].cla()
        axarr[row, col].imshow(batch.get_data()[i])
        axarr[row, col].set_title("[{}] {}\n[width: {:.1f} | height: {:.1f}]"
                                  .format(labels["set"], labels["file"], labels["width"], labels["height"]),
                                  fontsize=8)

        # Create a Rectangle patch
        rect = patches.Rectangle((labels["x1"], labels["y1"]),
                                 labels["x2"]-labels["x1"],
                                 labels["y2"]-labels["y1"],
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Draw center
        circ_r = patches.Circle((labels["center_x"], labels["center_y"]), radius=3, color="red", fill=True)
        circ_b = patches.Circle((labels["center_x"], labels["center_y"]), radius=5, color="black", fill=True)

        # Add the patch to the Axes
        axarr[row, col].add_patch(rect)
        axarr[row, col].add_patch(circ_b)
        axarr[row, col].add_patch(circ_r)

    plt.show()

def press(event):
    if event.key == "n":
        plot_batch()
    elif event.key == "e":
        fig.clear()
        plt.close()

try:
    fig, axarr = plt.subplots(3, 4)
    cig = fig.canvas.mpl_connect('key_press_event', press)
    plot_batch()
finally:
    del cig
    exit("shutting down...")


