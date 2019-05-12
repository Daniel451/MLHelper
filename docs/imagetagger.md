# How to Download ImageTagger Datasets

Despite manually downloading imagesets from [imagetagger.bit-bots.de](https://imagetagger.bit-bots.de),
MLHelper supports automatic downloading of imagesets, including whole collections like ``bitbots.TEST.CHALLENGE_2018``.

**Note**: for now, we are supporting selected collections of the Hamburg Bit-Bots datasets **only**.


### Downloading Datasets

Create a Python script and simply call the ``ImageTaggerAPI``:

```
from MLHelper.datasets import *

api = ImageTaggerAPI()
api.login(user, password)
api.download_imagesets(bitbots.BallDatasetHandler.TEST.WOLVES)
```
