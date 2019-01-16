# Datasets

### Structure

Each top-level dataset is a named submodule, e.g. `MLHelper.datasets.bitbots`. These Submodules have datasets, in the case of `bitbots` they are split into `TRAIN` and `TEST`.

##### Bit-Bots

```python
bitbots.TRAIN.LEIPZIG
bitbots.TRAIN.NAGOYA
bitbots.TRAIN.IRAN
bitbots.TRAIN.MONTREAL
bitbots.TRAIN.BITBOTSLAB
bitbots.TRAIN.CHALLENGE_2018
bitbots.TRAIN.ALL
bitbots.TEST.NAGOYA
bitbots.TEST.WOLVES
bitbots.TEST.IRAN
bitbots.TEST.BITBOTSLAB_CONCEALED
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
