import warnings

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn
import evalml.demos
import evalml.model_family
import evalml.objectives
import evalml.pipelines
import evalml.preprocessing
import evalml.problem_types
import evalml.utils
import evalml.data_checks
from evalml.automl import AutoMLSearch, search
from evalml.utils import print_info
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'The following selectors were not present in your DataTable')


__version__ = '0.24.1'
