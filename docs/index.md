# Documentation

MLHelper is a free-to-use pip module for machine learning experiments. 


### Requirements

 * OpenCV 3 or later
 * Python 3.6 or later


### Installation

You can install and update `MLHelper` directly via pip:

```bash
pip install MLHelper
```

### Structure

* [ImgReader](imgreader.md)
    * Class. Used to retrieve a data object and loading images into NumPy arrays **without** labels.
* [ImgLblReader](imglblreader.md)
    * Class. Used to retrieve a data object, loading images into NumPy arrays, and **parsing labels**.
* [datasets](datasets)
    * Submodule that holds different dataset objects, i.e. a collection of paths that can be passed to `ImgReader` and `ImgLblReader`.

