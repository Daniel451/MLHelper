# ImgReader

The `ImgReader` class is used to load images from various paths into NumPy arrays without considering any labels.

### Instantiate

```python
dat = ImgReader(pathlist: List[str],
                batch_size: int = 1,
                queue_size: int = 16,
                processes: int = None,
                img_dim: tuple = (200, 150),
                wait_for_queue_full = True,
                filter_labels = False)
```

* `pathlist`: a list of strings describing each path that `ImgReader` should check for images
* `batch_size`: images are automatically stored in batches with an alterable batch size
* `queue_size`: size of the internal multiprocessing queue; this setting affects memory usage
* `processes`: by default half of the available CPU cores are used, but you can define any number of worker processes manually
* `img_dim`: (width, height) to resize each image to
* `wait_for_queue_full`: whether or not the internal queue should block access before being filled for the first time
* `filter_labels`: deprecated; will be removed in a future release

### Methods

* `get_dataset_size()`: returns the total number of images in all given paths
* `get_max_queue_size()`: returns the queue's capacity
* `get_next_img_batch()`: pops the next image batch from queue; blocks automatically if queue is empty; each batch consists of a tuple: `(np.ndarray, list) -> (img_data, img_filepaths)`
