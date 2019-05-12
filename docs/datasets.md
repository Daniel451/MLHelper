# Datasets

### Download Datasets

An [ImageTagger](https://imagetagger.bit-bots.de) API was introduced in [v0.3.5](https://github.com/Daniel451/MLHelper/releases/tag/v0.3.5). See [How to Download ImageTagger Datasets](imagetagger.md) for more information.

### Structure

Each top-level dataset is a named submodule, e.g. `MLHelper.datasets.bitbots`. These Submodules have datasets, in the case of `bitbots` they are split into `TRAIN` and `TEST`.

##### Bit-Bots

In order for Bit-Bots datasets to work you need to set the environment variable `ROBO_AI_DATA`:

```export ROBO_AI_DATA=/path/to/root```

All Bit-Bots datasets need to reside in that directory.

Available ``TRAIN`` datasets:

```python
bitbots.TRAIN.LEIPZIG
bitbots.TRAIN.NAGOYA
bitbots.TRAIN.IRAN
bitbots.TRAIN.MONTREAL
bitbots.TRAIN.BITBOTSLAB
bitbots.TRAIN.CHALLENGE_2018
bitbots.TRAIN.ALL
```

Available ``TEST`` datasets:

```python
bitbots.TEST.NAGOYA
bitbots.TEST.WOLVES
bitbots.TEST.IRAN
bitbots.TEST.BITBOTSLAB_CONCEALED
bitbots.TEST.CHALLENGE_2018
bitbots.TEST.ALL
```


Available ``TEST_NOISED`` datasets (i.e. noised versions of test images):

```python
bitbots.TEST.NAGOYA
bitbots.TEST.WOLVES
bitbots.TEST.REAL
bitbots.TEST.CHALLENGE_2018
bitbots.TEST.ALL
```

For an explanation of `CHALLENGE_2018` see [Towards Real-Time Ball Localization using CNNs](https://robocup.informatik.uni-hamburg.de/wp-content/uploads/2018/06/2018_Speck_Ball_Localization.pdf) by Speck et al.

### Bit-Bots --- Example

```python
import MLHelper as H
from MLHelper.datasets.bitbots import BallDatasetHandler

# creates a dataset object with batch size 8 for alle datasets that were included in 'CHALLENGE 2018'
dat = H.ImgReader(BallDatasetHandler.TRAIN.CHALLENGE_2018, batch_size=8)
```
