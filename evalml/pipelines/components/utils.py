import inspect

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from evalml.exceptions import MissingComponentError
from evalml.model_family.utils import handle_model_family
from evalml.pipelines.components.component_base import ComponentBase
from evalml.pipelines.components.estimators.estimator import Estimator
from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    get_importable_subclasses,
    get_logger
)

logger = get_logger(__file__)


def _all_estimators():
    return get_importable_subclasses(Estimator, used_in_automl=False)


def _all_estimators_used_in_search():
    return get_importable_subclasses(Estimator, used_in_automl=True)


def _all_transformers():
    return get_importable_subclasses(Transformer, used_in_automl=False)


def all_components():
    return _all_estimators() + _all_transformers()


def allowed_model_families(problem_type):
    """List the model types allowed for a particular problem type.

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        list[ModelFamily]: a list of model families
    """

    estimators = []
    problem_type = handle_problem_types(problem_type)
    for estimator in _all_estimators_used_in_search():
        if problem_type in set(handle_problem_types(problem) for problem in estimator.supported_problem_types):
            estimators.append(estimator)

    return list(set([e.model_family for e in estimators]))


def get_estimators(problem_type, model_families=None):
    """Returns the estimators allowed for a particular problem type.

    Can also optionally filter by a list of model types.

    Arguments:
        problem_type (ProblemTypes or str): problem type to filter for
        model_families (list[ModelFamily] or list[str]): model families to filter for

    Returns:
        list[class]: a list of estimator subclasses
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")
    problem_type = handle_problem_types(problem_type)
    if model_families is None:
        model_families = allowed_model_families(problem_type)

    model_families = [handle_model_family(model_family) for model_family in model_families]
    all_model_families = allowed_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

    estimator_classes = []
    for estimator_class in _all_estimators_used_in_search():
        if problem_type not in [handle_problem_types(supported_pt) for supported_pt in estimator_class.supported_problem_types]:
            continue
        if estimator_class.model_family not in model_families:
            continue
        estimator_classes.append(estimator_class)
    return estimator_classes


def handle_component_class(component_class):
    """Standardizes input from a string name to a ComponentBase subclass if necessary.

    If a str is provided, will attempt to look up a ComponentBase class by that name and
    return a new instance. Otherwise if a ComponentBase subclass is provided, will return that
    without modification.

    Arguments:
        component (str, ComponentBase): input to be standardized

    Returns:
        ComponentBase
    """
    if inspect.isclass(component_class) and issubclass(component_class, ComponentBase):
        return component_class
    if not isinstance(component_class, str):
        raise ValueError(("component_graph may only contain str or ComponentBase subclasses, not '{}'")
                         .format(type(component_class)))
    component_classes = {component.name: component for component in all_components()}
    if component_class not in component_classes:
        raise MissingComponentError('Component "{}" was not found'.format(component_class))
    component_class = component_classes[component_class]
    return component_class


class WrappedSKClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn classifier wrapper class."""

    def __init__(self, pipeline):
        """Scikit-learn classifier wrapper class. Takes an EvalML pipeline as input
            and returns a scikit-learn classifier class wrapping that pipeline.

        Arguments:
            pipeline (PipelineBase or subclass obj): EvalML pipeline
        """
        self.pipeline = pipeline
        self._estimator_type = "classifier"
        if pipeline._is_fitted:
            self.feature_importances_ = pipeline.feature_importance
            self._is_fitted = True
            self.classes_ = pipeline.classes_

    def fit(self, X, y):
        """Fits component to data

        Arguments:
            X (pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self
        """
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Features

        Returns:
            np.ndarray: Predicted values
        """
        check_is_fitted(self, 'is_fitted_')

        return _convert_woodwork_types_wrapper(self.pipeline.predict(X).to_series()).to_numpy()

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Features

        Returns:
            np.ndarray: Probability estimates
        """
        return _convert_woodwork_types_wrapper(self.pipeline.predict_proba(X).to_dataframe()).to_numpy()


class WrappedSKRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn regressor wrapper class."""

    def __init__(self, pipeline):
        """Scikit-learn regressor wrapper class. Takes an EvalML pipeline as input
            and returns a scikit-learn regressor class wrapping that pipeline.

        Arguments:
            pipeline (PipelineBase or subclass obj): EvalML pipeline
        """
        self.pipeline = pipeline
        self._estimator_type = "regressor"
        if pipeline._is_fitted:
            self.feature_importances_ = pipeline.feature_importance
            self._is_fitted = True

    def fit(self, X, y):
        """Fits component to data

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Features

        Returns:
            np.ndarray: Predicted values
        """
        return self.pipeline.predict(X).to_series().to_numpy()


def scikit_learn_wrapped_estimator(evalml_obj):
    from evalml.pipelines.pipeline_base import PipelineBase

    """Wrap an EvalML pipeline or estimator in a scikit-learn estimator."""
    if isinstance(evalml_obj, PipelineBase):
        if evalml_obj.problem_type in [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]:
            return WrappedSKRegressor(evalml_obj)
        elif evalml_obj.problem_type == ProblemTypes.BINARY or evalml_obj.problem_type == ProblemTypes.MULTICLASS:
            return WrappedSKClassifier(evalml_obj)
    else:
        # EvalML Estimator
        if evalml_obj.supported_problem_types == [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]:
            return WrappedSKRegressor(evalml_obj)
        elif evalml_obj.supported_problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                                                    ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]:
            return WrappedSKClassifier(evalml_obj)
    raise ValueError("Could not wrap EvalML object in scikit-learn wrapper.")


def generate_component_code(element):
    """Creates and returns a string that contains the Python imports and code required for running the EvalML component.

    Arguments:
        element (component instance): The instance of the component to generate string Python code for

    Returns:
        String representation of Python code that can be run separately in order to recreate the component instance.
        Does not include code for custom component implementation.
    """
    # hold the imports needed and add code to end
    code_strings = []
    base_string = ""

    if not isinstance(element, ComponentBase):
        raise ValueError("Element must be a component instance, received {}".format(type(element)))

    if element.__class__ in all_components():
        code_strings.append("from {} import {}\n".format(element.__class__.__module__, element.__class__.__name__))
    component_parameters = element.parameters
    name = element.name[0].lower() + element.name[1:].replace(' ', '')
    base_string += "{0} = {1}(**{2})" \
                   .format(name,
                           element.__class__.__name__,
                           component_parameters)

    code_strings.append(base_string)
    return "\n".join(code_strings)


def make_balancing_dictionary(y, sampling_ratio):
    """Makes dictionary for oversampler components. Find ratio of each class to the majority.
    If the ratio is smaller than the sampling_ratio, we want to oversample,
    otherwise, we don't want to sample at all, and we leave the data as is.

    Arguments:
        y (pd.Series): Target data
        sampling_ratio (float): The balanced ratio we want the samples to meet

    Returns:
        Dictionary where keys are the classes, and the corresponding values are the counts of samples
        for each class that will satisfy sampling_ratio.
    """
    if sampling_ratio <= 0 or sampling_ratio > 1:
        raise ValueError("Sampling ratio must be in range (0, 1], received {}".format(sampling_ratio))
    if len(y) == 0:
        raise ValueError("Target data must not be empty")
    value_counts = y.value_counts()
    ratios = value_counts / value_counts.values[0]
    class_dic = {}
    sample_amount = int(value_counts.values[0] * sampling_ratio)
    for index, value in ratios.items():
        if value < sampling_ratio:
            # we want to oversample this class
            class_dic[index] = sample_amount
        else:
            # this class is already larger than the ratio, don't change
            class_dic[index] = value_counts[index]
    return class_dic
