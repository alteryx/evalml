"""EvalML."""
import warnings

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn
import evalml.demos
import evalml.model_family
import evalml.model_understanding
import evalml.objectives
import evalml.pipelines
import evalml.preprocessing
import evalml.problem_types
import evalml.utils
import evalml.data_checks
from evalml.automl import AutoMLSearch, search_iterative, search
from evalml.utils import print_info, update_checker

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

__version__ = "0.63.0"
