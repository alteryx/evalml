from collections import namedtuple

from evalml.objectives.utils import get_objective
from evalml.pipelines import BinaryClassificationPipeline
from evalml.preprocessing.data_splitters import TrainingValidationSplit


# Top-level replacement for AutoML object to supply data for testing purposes.
def err_call(*args, **kwargs):
    return 1


AutoMLSearchStruct = namedtuple("AutoML",
                                "data_splitter problem_type objective additional_objectives optimize_thresholds error_callback random_seed ensembling_indices")
data_splitter = TrainingValidationSplit()
problem_type = "binary"
objective = get_objective("Log Loss Binary", return_instance=True)
additional_objectives = []
optimize_thresholds = False
error_callback = err_call
random_seed = 0
ensembling_indices = [0]
automl_data = AutoMLSearchStruct(data_splitter, problem_type, objective, additional_objectives,
                                 optimize_thresholds, error_callback, random_seed, ensembling_indices)


class TestLRCPipeline(BinaryClassificationPipeline):
    component_graph = ["Logistic Regression Classifier"]


class TestSVMPipeline(BinaryClassificationPipeline):
    component_graph = ["SVM Classifier"]


class TestBaselinePipeline(BinaryClassificationPipeline):
    component_graph = ["Baseline Classifier"]
