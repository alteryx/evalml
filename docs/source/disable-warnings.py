# flake8: noqa 401 imported to force console mode for tqdm in jupyter notebooks
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings("ignore")
