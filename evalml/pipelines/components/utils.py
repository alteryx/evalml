"""Utility methods for EvalML components."""
import inspect
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from evalml.exceptions import MissingComponentError
from evalml.model_family.utils import ModelFamily, handle_model_family
from evalml.pipelines.components.component_base import ComponentBase
from evalml.pipelines.components.estimators.estimator import Estimator
from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_importable_subclasses


def _all_estimators():
    return get_importable_subclasses(Estimator, used_in_automl=False)


def _all_estimators_used_in_search():
    return get_importable_subclasses(Estimator, used_in_automl=True)


def _all_transformers():
    return get_importable_subclasses(Transformer, used_in_automl=False)


def all_components():
    """Get all available components."""
    return _all_estimators() + _all_transformers()


def allowed_model_families(problem_type):
    """List the model types allowed for a particular problem type.

    Args:
        problem_type (ProblemTypes or str): ProblemTypes enum or string.

    Returns:
        list[ModelFamily]: A list of model families.
    """
    estimators = []
    problem_type = handle_problem_types(problem_type)
    for estimator in _all_estimators_used_in_search():
        if problem_type in set(
            handle_problem_types(problem)
            for problem in estimator.supported_problem_types
        ):
            estimators.append(estimator)

    return list(set([e.model_family for e in estimators]))


def get_estimators(problem_type, model_families=None):
    """Returns the estimators allowed for a particular problem type.

    Can also optionally filter by a list of model types.

    Args:
        problem_type (ProblemTypes or str): Problem type to filter for.
        model_families (list[ModelFamily] or list[str]): Model families to filter for.

    Returns:
        list[class]: A list of estimator subclasses.

    Raises:
        TypeError: If the model_families parameter is not a list.
        RuntimeError: If a model family is not valid for the problem type.
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")
    problem_type = handle_problem_types(problem_type)
    if model_families is None:
        model_families = allowed_model_families(problem_type)

    model_families = [
        handle_model_family(model_family) for model_family in model_families
    ]
    all_model_families = allowed_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError(
                "Unrecognized model type for problem type %s: %s"
                % (problem_type, model_family),
            )

    estimator_classes = []
    for estimator_class in _all_estimators_used_in_search():
        if problem_type not in [
            handle_problem_types(supported_pt)
            for supported_pt in estimator_class.supported_problem_types
        ]:
            continue
        if estimator_class.model_family not in model_families:
            continue
        estimator_classes.append(estimator_class)
    return estimator_classes


def estimator_unable_to_handle_nans(estimator_class):
    """If True, provided estimator class is unable to handle NaN values as an input.

    Args:
        estimator_class (Estimator): Estimator class

    Raises:
        ValueError: If estimator is not a valid estimator class.

    Returns:
        bool: True if estimator class is unable to process NaN values, False otherwise.
    """
    if not hasattr(estimator_class, "model_family"):
        raise ValueError("`estimator_class` must have a `model_family` attribute.")
    return estimator_class.model_family in [
        ModelFamily.EXTRA_TREES,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.LINEAR_MODEL,
        ModelFamily.DECISION_TREE,
    ]


def handle_component_class(component_class):
    """Standardizes input from a string name to a ComponentBase subclass if necessary.

    If a str is provided, will attempt to look up a ComponentBase class by that name and
    return a new instance. Otherwise if a ComponentBase subclass or Component instance is provided,
    will return that without modification.

    Args:
        component_class (str, ComponentBase): Input to be standardized.

    Returns:
        ComponentBase

    Raises:
        ValueError: If input is not a valid component class.
        MissingComponentError: If the component cannot be found.

    Examples:
        >>> from evalml.pipelines.components.estimators.regressors.decision_tree_regressor import DecisionTreeRegressor
        >>> handle_component_class(DecisionTreeRegressor)
        <class 'evalml.pipelines.components.estimators.regressors.decision_tree_regressor.DecisionTreeRegressor'>
        >>> handle_component_class("Random Forest Regressor")
        <class 'evalml.pipelines.components.estimators.regressors.rf_regressor.RandomForestRegressor'>
    """
    if isinstance(component_class, ComponentBase) or (
        inspect.isclass(component_class) and issubclass(component_class, ComponentBase)
    ):
        return component_class
    if not isinstance(component_class, str):
        raise ValueError(
            (
                "component_class may only contain str or ComponentBase subclasses, not '{}'"
            ).format(type(component_class)),
        )
    component_classes = {component.name: component for component in all_components()}
    if component_class not in component_classes:
        raise MissingComponentError(
            'Component "{}" was not found'.format(component_class),
        )
    component_class = component_classes[component_class]
    return component_class


def get_prediction_intevals_for_tree_regressors(
    X: pd.DataFrame,
    predictions: pd.Series,
    coverage: List[float],
    estimators: List[Estimator],
) -> Dict[str, pd.Series]:
    """Find the prediction intervals for tree-based regressors.

    Args:
        X (pd.DataFrame): Data of shape [n_samples, n_features].
        predictions (pd.Series): Predictions from the regressor.
        coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
            prediction interval should be calculated for.
        estimators (list): Collection of fitted sub-estimators.

    Returns:
        dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
    """
    prediction_interval_result = {}
    for conf_int in coverage:
        preds = np.zeros((len(X), len(estimators)))
        for ind, estimator_ in enumerate(estimators):
            preds[:, ind] = estimator_.predict(X)
        std_preds = np.std(preds, axis=1)
        preds_lower = (
            predictions + st.norm.ppf(round((1 - conf_int) / 2, 3)) * std_preds
        )
        preds_upper = (
            predictions + st.norm.ppf(round((1 + conf_int) / 2, 3)) * std_preds
        )

        preds_lower = pd.Series(preds_lower, index=X.index, name=None)
        preds_upper = pd.Series(preds_upper, index=X.index, name=None)
        prediction_interval_result[f"{conf_int}_lower"] = preds_lower
        prediction_interval_result[f"{conf_int}_upper"] = preds_upper

    return prediction_interval_result


class WrappedSKClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn classifier wrapper class."""

    def __init__(self, pipeline):
        """Scikit-learn classifier wrapper class. Takes an EvalML pipeline as input and returns a scikit-learn classifier class wrapping that pipeline.

        Args:
            pipeline (PipelineBase or subclass obj): EvalML pipeline.
        """
        self.pipeline = pipeline
        self._estimator_type = "classifier"
        if pipeline._is_fitted:
            self._is_fitted = True
            self.classes_ = pipeline.classes_

    def fit(self, X, y):
        """Fits component to data.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

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

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.ndarray: Predicted values.
        """
        check_is_fitted(self, "is_fitted_")

        return self.pipeline.predict(X).to_numpy()

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.ndarray: Probability estimates.
        """
        return self.pipeline.predict_proba(X).to_numpy()


class WrappedSKRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn regressor wrapper class."""

    def __init__(self, pipeline):
        """Scikit-learn regressor wrapper class. Takes an EvalML pipeline as input and returns a scikit-learn regressor class wrapping that pipeline.

        Args:
            pipeline (PipelineBase or subclass obj): EvalML pipeline.
        """
        self.pipeline = pipeline
        self._estimator_type = "regressor"
        self._is_fitted_ = True  # We need an attribute that ends in an underscore for scikit-learn to treat as fitted

    def fit(self, X, y):
        """Fits component to data.

        Args:
            X (pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.pipeline.predict(X).to_numpy()


def scikit_learn_wrapped_estimator(evalml_obj):
    """Wraps an EvalML object as a scikit-learn estimator."""
    from evalml.pipelines.pipeline_base import PipelineBase

    """Wrap an EvalML pipeline or estimator in a scikit-learn estimator."""
    if isinstance(evalml_obj, PipelineBase):
        if evalml_obj.problem_type in [
            ProblemTypes.REGRESSION,
            ProblemTypes.TIME_SERIES_REGRESSION,
        ]:
            return WrappedSKRegressor(evalml_obj)
        elif (
            evalml_obj.problem_type == ProblemTypes.BINARY
            or evalml_obj.problem_type == ProblemTypes.MULTICLASS
        ):
            return WrappedSKClassifier(evalml_obj)
    else:
        # EvalML Estimator
        if evalml_obj.supported_problem_types == [
            ProblemTypes.REGRESSION,
            ProblemTypes.TIME_SERIES_REGRESSION,
        ]:
            return WrappedSKRegressor(evalml_obj)
        elif evalml_obj.supported_problem_types == [
            ProblemTypes.BINARY,
            ProblemTypes.MULTICLASS,
            ProblemTypes.TIME_SERIES_BINARY,
            ProblemTypes.TIME_SERIES_MULTICLASS,
        ]:
            return WrappedSKClassifier(evalml_obj)
    raise ValueError("Could not wrap EvalML object in scikit-learn wrapper.")


def generate_component_code(element):
    r"""Creates and returns a string that contains the Python imports and code required for running the EvalML component.

    Args:
        element (component instance): The instance of the component to generate string Python code for.

    Returns:
        String representation of Python code that can be run separately in order to recreate the component instance.
        Does not include code for custom component implementation.

    Raises:
        ValueError: If the input element is not a component instance.

    Examples:
        >>> from evalml.pipelines.components.estimators.regressors.decision_tree_regressor import DecisionTreeRegressor
        >>> assert generate_component_code(DecisionTreeRegressor()) == "from evalml.pipelines.components.estimators.regressors.decision_tree_regressor import DecisionTreeRegressor\n\ndecisionTreeRegressor = DecisionTreeRegressor(**{'criterion': 'squared_error', 'max_features': 'auto', 'max_depth': 6, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0})"
        ...
        >>> from evalml.pipelines.components.transformers.imputers.simple_imputer import SimpleImputer
        >>> assert generate_component_code(SimpleImputer()) == "from evalml.pipelines.components.transformers.imputers.simple_imputer import SimpleImputer\n\nsimpleImputer = SimpleImputer(**{'impute_strategy': 'most_frequent', 'fill_value': None})"
    """
    # hold the imports needed and add code to end
    code_strings = []
    base_string = ""

    if not isinstance(element, ComponentBase):
        raise ValueError(
            "Element must be a component instance, received {}".format(type(element)),
        )

    if element.__class__ in all_components():
        code_strings.append(
            "from {} import {}\n".format(
                element.__class__.__module__,
                element.__class__.__name__,
            ),
        )
    component_parameters = element.parameters
    name = element.name[0].lower() + element.name[1:].replace(" ", "")
    base_string += "{0} = {1}(**{2})".format(
        name,
        element.__class__.__name__,
        component_parameters,
    )

    code_strings.append(base_string)
    return "\n".join(code_strings)


def make_balancing_dictionary(y, sampling_ratio):
    """Makes dictionary for oversampler components. Find ratio of each class to the majority. If the ratio is smaller than the sampling_ratio, we want to oversample, otherwise, we don't want to sample at all, and we leave the data as is.

    Args:
        y (pd.Series): Target data.
        sampling_ratio (float): The balanced ratio we want the samples to meet.

    Returns:
        dict: Dictionary where keys are the classes, and the corresponding values are the counts of samples
        for each class that will satisfy sampling_ratio.

    Raises:
        ValueError: If sampling ratio is not in the range (0, 1] or the target is empty.

    Examples:
        >>> import pandas as pd
        >>> y = pd.Series([1] * 4 + [2] * 8 + [3])
        >>> assert make_balancing_dictionary(y, 0.5) == {2: 8, 1: 4, 3: 4}
        >>> assert make_balancing_dictionary(y, 0.9) == {2: 8, 1: 7, 3: 7}
        >>> assert make_balancing_dictionary(y, 0.1) == {2: 8, 1: 4, 3: 1}
    """
    if sampling_ratio <= 0 or sampling_ratio > 1:
        raise ValueError(
            "Sampling ratio must be in range (0, 1], received {}".format(
                sampling_ratio,
            ),
        )
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


def handle_float_categories_for_catboost(X):
    """Updates input data to be compatible with CatBoost estimators.

    CatBoost cannot handle data in X that is the Categorical Woodwork logical type with floating point categories.
    This utility determines if the floating point categories can be converted to integers
    without truncating any data, and if they can be, converts them to int64 categories.
    Will not attempt to use values that are truly floating points.

    Args:
        X (pd.DataFrame): Input data to CatBoost that has Woodwork initialized

    Returns:
        DataFrame: Input data with exact same Woodwork typing info as the original but with any float categories
            converted to be int64 when possible.

    Raises:
        ValueError: if the numeric categories are actual floats that cannot be converted to integers
            without truncating data
    """
    original_schema = X.ww.schema
    original_dtypes = X.dtypes

    # Determine which categorical columns have float categories, which CatBoost would error on
    categorical_columns = X.ww.select("category", return_schema=True).columns.keys()
    cols_with_float_categories = [
        col
        for col in categorical_columns
        if original_dtypes[col].categories.dtype == "float64"
    ]

    if not cols_with_float_categories:
        return X

    # determine which columns are really integers vs are actually floats
    new_dtypes = {}
    for col in cols_with_float_categories:
        col_categories = original_dtypes[col].categories
        floats_are_really_ints = (col_categories % 1 == 0).all()
        if floats_are_really_ints:
            # We can use non nullable int64 here because there will not be any nans at this point
            new_categories = col_categories.astype("int64")
            new_dtypes[col] = pd.CategoricalDtype(
                categories=new_categories,
                ordered=original_dtypes[col].ordered,
            )
        else:
            # CatBoost explanation as to why they don't support float categories: https://catboost.ai/en/docs/concepts/faq#floating-point-values
            # CatBoost bug keeping us from converting to string: https://github.com/catboost/catboost/issues/1965
            # Pandas bug keeping us from converting `.astype("string").astype("object")`: https://github.com/pandas-dev/pandas/issues/51074
            raise ValueError(
                f"Invalid category found in {col}. CatBoost does not support floats as categories.",
            )

    X_t = X.astype(new_dtypes)
    X_t.ww.init(schema=original_schema)
    return X_t
