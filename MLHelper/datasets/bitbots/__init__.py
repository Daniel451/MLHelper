from ._bitbots import BallDatasetHandler
from ._bitbots import NegativeBallDatasetHandler
from ._bitbots import RandomDatasetHandler
from ._bitbots import ImagesetCollection

import os

if "ROBO_AI_DATA" not in os.environ:
    raise Exception("environment variable 'ROBO_AI_DATA' is not set."
                    + "\n\nset this variable with 'export ROBO_AI_DATA=/path/to/dir'")
elif os.environ["ROBO_AI_DATA"] == "":
    raise Exception("environment variable 'ROBO_AI_DATA' is empty."
                    + "\n\nset this variable with 'export ROBO_AI_DATA=/path/to/dir'")


