# flake8:noqa

import warnings

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt

import evalml.demos
import evalml.model_types
import evalml.objectives
import evalml.pipelines
# import evalml.models
import evalml.preprocessing
import evalml.problem_types
import evalml.utils
import evalml.guardrails

from evalml.pipelines import list_model_types, save_pipeline, load_pipeline
from evalml.models import AutoClassifier, AutoRegressor
from evalml.pipelines import list_model_types, load_pipeline, save_pipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)


__version__ = '0.5.1'
