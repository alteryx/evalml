import pandas as pd

from .binary_classification_pipeline import BinaryClassificationPipeline
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MetaEstimatorMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .regression_pipeline import RegressionPipeline

from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    CatBoostClassifier,
    CatBoostRegressor,
    DateTimeFeaturizer,
    DropNullColumns,
    Estimator,
    Imputer,
    OneHotEncoder,
    StandardScaler
)
from evalml.pipelines.components.utils import get_estimators
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger
from evalml.utils.gen_utils import categorical_dtypes, datetime_dtypes

logger = get_logger(__file__)


def _get_preprocessing_components(X, y, problem_type, estimator_class):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Arguments:
        X (pd.DataFrame): The input data of shape [n_samples, n_features]
        y (pd.Series): The target data of length [n_samples]
        problem_type (ProblemTypes or str): Problem type
        estimator_class (class): A class which subclasses Estimator estimator for pipeline

    Returns:
        list[Transformer]: A list of applicable preprocessing components to use with the estimator
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    pp_components = []
    all_null_cols = X.columns[X.isnull().all()]
    if len(all_null_cols) > 0:
        pp_components.append(DropNullColumns)

    pp_components.append(Imputer)

    datetime_cols = X.select_dtypes(include=datetime_dtypes)
    add_datetime_featurizer = len(datetime_cols.columns) > 0
    if add_datetime_featurizer:
        pp_components.append(DateTimeFeaturizer)

    # DateTimeFeaturizer can create categorical columns
    categorical_cols = X.select_dtypes(include=categorical_dtypes)
    if (add_datetime_featurizer or len(categorical_cols.columns) > 0) and estimator_class not in {CatBoostClassifier, CatBoostRegressor}:
        pp_components.append(OneHotEncoder)

    if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
        pp_components.append(StandardScaler)
    return pp_components


def _get_pipeline_base_class(problem_type):
    """Returns pipeline base class for problem_type"""
    if problem_type == ProblemTypes.BINARY:
        return BinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        return MulticlassClassificationPipeline
    elif problem_type == ProblemTypes.REGRESSION:
        return RegressionPipeline


def make_pipeline(X, y, estimator, problem_type):
    """Given input data, target data, an estimator class and the problem type,
        generates a pipeline class with a preprocessing chain which was recommended based on the inputs.
        The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type.

   Arguments:
        X (pd.DataFrame): The input data of shape [n_samples, n_features]
        y (pd.Series): The target data of length [n_samples]
        estimator (Estimator): Estimator for pipeline
        problem_type (ProblemTypes or str): Problem type for pipeline to generate

    Returns:
        class: PipelineBase subclass with dynamically generated preprocessing components and specified estimator

    """
    problem_type = handle_problem_types(problem_type)
    if estimator not in get_estimators(problem_type):
        raise ValueError(f"{estimator.name} is not a valid estimator for problem type")
    preprocessing_components = _get_preprocessing_components(X, y, problem_type, estimator)
    complete_component_graph = preprocessing_components + [estimator]

    hyperparameters = None
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    base_class = _get_pipeline_base_class(problem_type)

    class GeneratedPipeline(base_class):
        custom_name = f"{estimator.name} w/ {' + '.join([component.name for component in preprocessing_components])}"
        component_graph = complete_component_graph
        custom_hyperparameters = hyperparameters

    return GeneratedPipeline


def make_pipeline_from_components(component_instances, problem_type, custom_name=None):
    """Given a list of component instances and the problem type, a pipeline instance is generated with the component instances.
    The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type. A custom name for
    the pipeline can optionally be specified; otherwise the default pipeline name will be 'Templated Pipeline'.

   Arguments:
        component_instances (list): a list of all of the components to include in the pipeline
        problem_type (str or ProblemTypes): problem type for the pipeline to generate
        custom_name (string): a name for the new pipeline

    Returns:
        Pipeline instance with component instances and specified estimator

    """
    if not isinstance(component_instances[-1], Estimator):
        raise ValueError("Pipeline needs to have an estimator at the last position of the component list")

    pipeline_name = custom_name
    problem_type = handle_problem_types(problem_type)

    class TemplatedPipeline(_get_pipeline_base_class(problem_type)):
        custom_name = pipeline_name
        component_graph = [c.__class__ for c in component_instances]
        # def __reduce__(self):
        #     # return a class which can return this class when called with the 
        #     # appropriate tuple of arguments
        #     return (TemplatedPipeline, (self.parameters, self.random_state))
        
    TemplatedPipeline = globals()["TemplatedPipeline"]
    pipeline_instance = TemplatedPipeline({})
    pipeline_instance.component_graph = component_instances
    return pipeline_instance


class WrappedSKClassifier(ClassifierMixin, BaseEnsemble):

    requires_fit = True
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline_ = self.pipeline.clone()
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return self.pipeline_.predict(X).to_numpy()

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')
        return self.pipeline_.predict_proba(X).to_numpy()


    def get_params(self, deep=True):
        return {"pipeline": self.pipeline}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class WrappedSKRegressor(RegressorMixin, MetaEstimatorMixin, BaseEstimator):

    def __init__(self, pipeline):
        print ("HELLO")
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline_ = self.pipeline.clone()

        X, y = check_X_y(X, y)
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True

        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return self.pipeline_.predict(X).to_numpy()

    def get_params(self, deep=True):
        return {"pipeline": self.pipeline}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


from evalml.pipelines.components import RandomForestClassifier, RandomForestRegressor
# class MockPipelineB(BinaryClassificationPipeline):
#     component_graph = ['Imputer', 'Random Forest Classifier']
# class MockPipelineM(MulticlassClassificationPipeline):
#     component_graph = ['Imputer', 'Random Forest Classifier']
# class MockPipelineR(RegressionPipeline):
#     component_graph = ['Imputer', 'Random Forest Regressor']

def scikit_learn_wrapped_estimator(evalml_obj, is_pipeline=True):
    """Wrap an EvalML pipeline or estimator in a scikit-learn estimator."""
    if is_pipeline:
        if evalml_obj.problem_type == ProblemTypes.REGRESSION:
            evalml_obj = RandomForestRegressor()
            return WrappedSKRegressor(evalml_obj)
        elif evalml_obj.problem_type == ProblemTypes.BINARY:
            # evalml_obj = MockPipelineB(parameters={})
            evalml_obj = RandomForestClassifier()
            return WrappedSKClassifier(evalml_obj)        
        elif evalml_obj.problem_type == ProblemTypes.MULTICLASS:
            evalml_obj = RandomForestClassifier()
            # evalml_obj = MockPipelineM(parameters={})
            return WrappedSKClassifier(evalml_obj)
    else:
        # EvalML Estimator
        if evalml_obj.supported_problem_types == [ProblemTypes.REGRESSION]:
            return WrappedSKRegressor(evalml_obj)
        elif evalml_obj.supported_problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
            return WrappedSKClassifier(evalml_obj)
    raise ValueError("Could not wrap EvalML object in scikit-learn wrapper.")
