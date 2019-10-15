# flake8:noqa

import warnings

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt

import evalml.demos
import evalml.objectives
import evalml.pipelines
# import evalml.models
import evalml.preprocessing
import evalml.problem_types
import evalml.model_types
import evalml.utils

from evalml.pipelines import list_model_types, save_pipeline, load_pipeline
from evalml.models import AutoClassifier, AutoRegressor

warnings.filterwarnings("ignore", category=DeprecationWarning)


__version__ = '0.4.1'
