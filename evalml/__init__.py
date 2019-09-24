# flake8:noqa

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import evalml.models
import evalml.preprocessing
import evalml.objectives
import evalml.tuners
import evalml.demos
import evalml.problem_types
import evalml.pipelines

from evalml.pipelines import list_model_types, save_pipeline, load_pipeline
from evalml.models import AutoClassifier, AutoRegressor

__version__ = '0.4.1'
