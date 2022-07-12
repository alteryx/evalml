"""Data splitter classes."""
from evalml.preprocessing.data_splitters.no_split import NoSplit
from evalml.preprocessing.data_splitters.training_validation_split import (
    TrainingValidationSplit,
)
from evalml.preprocessing.data_splitters.time_series_split import TimeSeriesSplit
from evalml.preprocessing.data_splitters.sk_splitters import KFold, StratifiedKFold
