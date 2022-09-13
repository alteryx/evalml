import time

from evalml.automl.utils import AutoMLConfig
from evalml.exceptions import PipelineScoreError
from evalml.objectives.utils import get_objective
from evalml.pipelines import BinaryClassificationPipeline
from evalml.preprocessing.data_splitters import TrainingValidationSplit


# Top-level replacement for AutoML object to supply data for testing purposes.
def err_call(*args, **kwargs):
    """No-op"""


data_splitter = TrainingValidationSplit()
problem_type = "binary"
objective = get_objective("Log Loss Binary", return_instance=True)
alternate_thresholding_objective = get_objective("F1", return_instance=True)
additional_objectives = []
optimize_thresholds = False
error_callback = err_call
random_seed = 0
automl_data = AutoMLConfig(
    data_splitter=data_splitter,
    problem_type=problem_type,
    objective=objective,
    additional_objectives=additional_objectives,
    optimize_thresholds=optimize_thresholds,
    alternate_thresholding_objective=alternate_thresholding_objective,
    error_callback=error_callback,
    random_seed=random_seed,
    X_schema=None,
    y_schema=None,
    errors={},
)


def delayed(delay):
    """Decorator to delay function evaluation."""

    def wrap(a_method):
        def do_delay(*args, **kw):
            time.sleep(delay)
            return a_method(*args, **kw)

        return do_delay

    return wrap


class DaskPipelineWithFitError(BinaryClassificationPipeline):
    component_graph = ["Baseline Classifier"]
    custom_name = "PipelineWithError"

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            custom_name=self.custom_name,
            random_seed=random_seed,
        )

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)

    @delayed(5)
    def fit(self, X, y):
        raise Exception("Yikes")


class DaskPipelineWithScoreError(BinaryClassificationPipeline):
    component_graph = ["Baseline Classifier"]
    custom_name = "PipelineWithError"

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            custom_name=self.custom_name,
            random_seed=random_seed,
        )

    def score(self, X, y, objectives, X_train=None, y_train=None):
        raise PipelineScoreError(
            exceptions={"AUC": (Exception(), []), "Log Loss Binary": (Exception(), [])},
            scored_successfully={
                "F1": 0.2,
                "MCC Binary": 0.2,
                "Precision": 0.8,
                "Balanced Accuracy Binary": 0.2,
                "Accuracy Binary": 0.2,
            },
        )


class DaskPipelineSlow(BinaryClassificationPipeline):
    """Pipeline for testing whose fit() should take longer than the
    fast pipeline.  This exists solely to test AutoMLSearch termination
    and not complete fitting."""

    component_graph = ["Baseline Classifier"]
    custom_name = "SlowPipeline"

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            custom_name=self.custom_name,
            random_seed=random_seed,
        )

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)

    @delayed(2)
    def fit(self, X, y):
        super().fit(X, y)


class DaskPipelineFast(BinaryClassificationPipeline):
    """Pipeline for testing whose fit() should complete before the
    slow pipeline.  This exists solely to test AutoMLSearch termination
    and complete fitting."""

    component_graph = ["Baseline Classifier"]
    custom_name = "FastPipeline"

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            custom_name=self.custom_name,
            random_seed=random_seed,
        )

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)

    def fit(self, X, y):
        self._is_fitted = True
        super().fit(X, y)


class DaskSchemaCheckPipeline(BinaryClassificationPipeline):
    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
        X_schema_to_check=None,
        y_schema_to_check=None,
    ):
        self.X_schema_to_check = X_schema_to_check
        self.y_schema_to_check = y_schema_to_check
        super().__init__(
            component_graph,
            parameters,
            custom_name,
            random_seed,
        )

    def clone(self):
        return self.__class__(
            self.component_graph,
            parameters=self.parameters,
            custom_name=self.custom_name,
            random_seed=self.random_seed,
            X_schema_to_check=self.X_schema_to_check,
            y_schema_to_check=self.y_schema_to_check,
        )

    def fit(self, X, y):
        assert X.ww.schema == self.X_schema_to_check
        assert y.ww.schema == self.y_schema_to_check
        return super().fit(X, y)

    def score(self, X, y, objectives, X_train=None, y_train=None):
        assert X.ww.schema == self.X_schema_to_check
        assert y.ww.schema == self.y_schema_to_check
        return super().score(X, y, objectives)
