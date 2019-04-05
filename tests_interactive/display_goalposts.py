import os
import MLHelper as H
from MLHelper.datasets.bitbots import BallDatasetHandler

from matplotlib import pyplot as plt
from matplotlib import patches


for img_dim in [(800, 600), (200, 150)]:
    dat = H.ImgLblReader([os.path.join(os.environ["ROBO_AI_DATA"], "bitbots-set00-01")],
                         batch_size=4,
                         queue_size=4,
                         img_dim=img_dim,
                         label_content="goalpost",
                         filter_labels=True)

    iterations = 5
    for i in range(iterations):
        # retrieve data
        batch = dat.get_next_batch()
        imgs = batch.get_data()
        labels = batch.get_labels()

        plt.title(f"img_dim {img_dim} iteration {i+1}/{iterations}")
        for p, j in zip(range(1, 5), range(4)):
            # active subplot p
            ax = plt.subplot(2, 2, p)
            plt.imshow(imgs[j], interpolation="None")

            # labels
            for label in labels[j]:
                rect = patches.Rectangle((label.x1, label.y1), label.width, label.height,
                                         linewidth=2, edgecolor="red", fill=False)
                ax.add_patch(rect)

        # show final image
        plt.show()
        plt.clf()
    plt.close()
