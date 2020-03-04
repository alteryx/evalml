# flake8:noqa

import warnings

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt

import evalml.demos
import evalml.model_family
import evalml.objectives
import evalml.pipelines
import evalml.preprocessing
import evalml.problem_types
import evalml.utils
import evalml.guardrails

from evalml.pipelines import list_model_families, save_pipeline, load_pipeline
from evalml.automl import AutoClassificationSearch, AutoRegressionSearch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


__version__ = '0.6.0'
