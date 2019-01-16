# ImgLblReader

The `ImgLblReader` class is used to load images from various paths into NumPy arrays and parsing their labels. Currently, the `ImgLblReader` class creates an iterator to iterate over all images in a random order and shuffles its internal order after visiting each image exactly once. Future release will cover more features.

At the moment, `ImgLblReader` **only** parses ball annotations in 'AI Ball' format of the [Bit-Bots ImageTagger](https://imagetagger.bit-bots.de).

### Instantiate

```python
dat = ImgLblReader(pathlist: List[str],
                   batch_size: int = 1,
                   queue_size: int = 16,
                   processes: int = None,
                   img_dim: tuple = (200, 150),
                   filter_labels = False)
```

* `pathlist`: a list of strings describing each path that `ImgReader` should check for images
* `batch_size`: images are automatically stored in batches with an alterable batch size
* `queue_size`: size of the internal multiprocessing queue; this setting affects memory usage
* `processes`: by default half of the available CPU cores are used, but you can define any number of worker processes manually
* `img_dim`: (width, height) to resize each image to
* `filter_labels`: deprecated; will be removed in a future release

### Methods

* `get_dataset_size()`: returns the total number of images in all given paths
* `get_max_queue_size()`: returns the queue's capacity
* `get_next_batch()`: pops the next `ImageBatch` object from queue; blocks automatically if queue is empty; each `ImageBatch` contains the original images and the corresponding labels
