"""Helpful preprocessing utilities."""
import logging

import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from woodwork import logical_types

from evalml.model_family import ModelFamily
from evalml.pipelines import (
    CatBoostClassifier,
    CatBoostRegressor,
    DropNaNRowsTransformer,
    OneHotEncoder,
    StandardScaler,
    TimeSeriesFeaturizer,
)
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    DropColumns,
    DropNullColumns,
    EmailFeaturizer,
    Imputer,
    LabelEncoder,
    NaturalLanguageFeaturizer,
    Oversampler,
    ReplaceNullableTypes,
    Undersampler,
    URLFeaturizer,
)
from evalml.pipelines.components.utils import estimator_unable_to_handle_nans
from evalml.preprocessing.data_splitters import TrainingValidationSplit
from evalml.problem_types import (
    is_classification,
    is_regression,
    is_time_series,
)
from evalml.utils import import_or_raise, infer_feature_types

logger = logging.getLogger(__name__)


def load_data(path, index, target, n_rows=None, drop=None, verbose=True, **kwargs):
    """Load features and target from file.

    Args:
        path (str): Path to file or a http/ftp/s3 URL.
        index (str): Column for index.
        target (str): Column for target.
        n_rows (int): Number of rows to return. Defaults to None.
        drop (list): List of columns to drop. Defaults to None.
        verbose (bool): If True, prints information about features and target. Defaults to True.
        **kwargs: Other keyword arguments that should be passed to panda's `read_csv` method.

    Returns:
        pd.DataFrame, pd.Series: Features matrix and target.
    """
    feature_matrix = pd.read_csv(path, index_col=index, nrows=n_rows, **kwargs)

    targets = [target] + (drop or [])
    y = feature_matrix[target]
    X = feature_matrix.drop(columns=targets)

    if verbose:
        # number of features
        print(number_of_features(X.dtypes), end="\n\n")

        # number of total training examples
        info = "Number of training examples: {}"
        print(info.format(len(X)), end="\n")

        # target distribution
        print(target_distribution(y))

    return infer_feature_types(X), infer_feature_types(y)


def split_data(
    X, y, problem_type, problem_configuration=None, test_size=0.2, random_seed=0
):
    """Split data into train and test sets.

    Args:
        X (pd.DataFrame or np.ndarray): data of shape [n_samples, n_features]
        y (pd.Series, or np.ndarray): target data of length [n_samples]
        problem_type (str or ProblemTypes): type of supervised learning problem. see evalml.problem_types.problemtype.all_problem_types for a full list.
        problem_configuration (dict): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the time_index, gap, and max_delay variables.
        test_size (float): What percentage of data points should be included in the test set. Defaults to 0.2 (20%).
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Feature and target data each split into train and test sets.

    Examples:
        >>> X = pd.DataFrame([1, 2, 3, 4, 5, 6], columns=["First"])
        >>> y = pd.Series([8, 9, 10, 11, 12, 13])
        ...
        >>> X_train, X_validation, y_train, y_validation = split_data(X, y, "regression", random_seed=42)
        >>> X_train
           First
        5      6
        2      3
        4      5
        3      4
        >>> X_validation
           First
        0      1
        1      2
        >>> y_train
        5    13
        2    10
        4    12
        3    11
        dtype: int64
        >>> y_validation
        0    8
        1    9
        dtype: int64
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    data_splitter = None
    if is_time_series(problem_type):
        data_splitter = TrainingValidationSplit(
            test_size=test_size, shuffle=False, stratify=None, random_seed=random_seed
        )
    elif is_regression(problem_type):
        data_splitter = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )
    elif is_classification(problem_type):
        data_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )

    train, test = next(data_splitter.split(X, y))

    X_train = X.ww.iloc[train]
    X_test = X.ww.iloc[test]
    y_train = y.ww.iloc[train]
    y_test = y.ww.iloc[test]

    return X_train, X_test, y_train, y_test


def number_of_features(dtypes):
    """Get the number of features of each specific dtype in a DataFrame.

    Args:
        dtypes (pd.Series): DataFrame.dtypes to get the number of features for.

    Returns:
        pd.Series: dtypes and the number of features for each input type.

    Example:
        >>> X = pd.DataFrame()
        >>> X["integers"] = [i for i in range(10)]
        >>> X["floats"] = [float(i) for i in range(10)]
        >>> X["strings"] = [str(i) for i in range(10)]
        >>> X["booleans"] = [bool(i%2) for i in range(10)]

        Lists the number of columns corresponding to each dtype.

        >>> number_of_features(X.dtypes)
                     Number of Features
        Boolean                       1
        Categorical                   1
        Numeric                       2
    """
    dtype_to_vtype = {
        "bool": "Boolean",
        "int32": "Numeric",
        "int64": "Numeric",
        "float64": "Numeric",
        "object": "Categorical",
        "datetime64[ns]": "Datetime",
    }

    vtypes = dtypes.astype(str).map(dtype_to_vtype).value_counts()
    return vtypes.sort_index().to_frame("Number of Features")


def target_distribution(targets):
    """Get the target distributions.

    Args:
        targets (pd.Series): Target data.

    Returns:
        pd.Series: Target data and their frequency distribution as percentages.

    Examples:
        >>> y = pd.Series([1, 2, 4, 1, 3, 3, 1, 2])
        >>> target_distribution(y)
        Targets
        1    37.50%
        2    25.00%
        3    25.00%
        4    12.50%
        dtype: object
        >>> y = pd.Series([True, False, False, False, True])
        >>> target_distribution(y)
        Targets
        False    60.00%
        True     40.00%
        dtype: object
    """
    distribution = targets.value_counts() / len(targets)
    return distribution.mul(100).apply("{:.2f}%".format).rename_axis("Targets")


def _get_preprocessing_components(
    X, y, problem_type, estimator_class, sampler_name=None
):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Args:
        X (pd.DataFrame): The input data of shape [n_samples, n_features].
        y (pd.Series): The target data of length [n_samples].
        problem_type (ProblemTypes or str): Problem type.
        estimator_class (class): A class which subclasses Estimator estimator for pipeline.
        sampler_name (str): The name of the sampler component to add to the pipeline. Defaults to None.

    Returns:
        list[Transformer]: A list of applicable preprocessing components to use with the estimator.
    """
    if is_time_series(problem_type):
        components_functions = [
            _get_label_encoder,
            _get_drop_all_null,
            _get_replace_null,
            _get_drop_index_unknown,
            _get_url_email,
            _get_natural_language,
            _get_imputer,
            _get_time_series_featurizer,
            _get_datetime,
            _get_ohe,
            _get_sampler,
            _get_standard_scaler,
            _get_drop_nan_rows_transformer,
        ]
    else:
        components_functions = [
            _get_label_encoder,
            _get_drop_all_null,
            _get_replace_null,
            _get_drop_index_unknown,
            _get_url_email,
            _get_datetime,
            _get_natural_language,
            _get_imputer,
            _get_ohe,
            _get_sampler,
            _get_standard_scaler,
        ]
    components = []
    for function in components_functions:
        components.extend(function(X, y, problem_type, estimator_class, sampler_name))

    return components


def _get_label_encoder(X, y, problem_type, estimator_class, sampler_name=None):
    component = []
    if is_classification(problem_type):
        component.append(LabelEncoder)
    return component


def _get_drop_all_null(X, y, problem_type, estimator_class, sampler_name=None):
    component = []
    non_index_unknown = X.ww.select(exclude=["index", "unknown"])
    all_null_cols = non_index_unknown.columns[non_index_unknown.isnull().all()]
    if len(all_null_cols) > 0:
        component.append(DropNullColumns)
    return component


def _get_replace_null(X, y, problem_type, estimator_class, sampler_name=None):
    component = []
    all_nullable_cols = X.ww.select(
        ["IntegerNullable", "AgeNullable", "BooleanNullable"], return_schema=True
    ).columns
    nullable_target = isinstance(
        y.ww.logical_type,
        (
            logical_types.AgeNullable,
            logical_types.BooleanNullable,
            logical_types.IntegerNullable,
        ),
    )
    if len(all_nullable_cols) > 0 or nullable_target:
        component.append(ReplaceNullableTypes)
    return component


def _get_drop_index_unknown(X, y, problem_type, estimator_class, sampler_name=None):
    component = []
    index_and_unknown_columns = list(
        X.ww.select(["index", "unknown"], return_schema=True).columns
    )
    if len(index_and_unknown_columns) > 0:
        component.append(DropColumns)
    return component


def _get_url_email(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    email_columns = list(X.ww.select("EmailAddress", return_schema=True).columns)
    if len(email_columns) > 0:
        components.append(EmailFeaturizer)

    url_columns = list(X.ww.select("URL", return_schema=True).columns)
    if len(url_columns) > 0:
        components.append(URLFeaturizer)

    return components


def _get_datetime(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    datetime_cols = list(X.ww.select(["Datetime"], return_schema=True).columns)

    add_datetime_featurizer = len(datetime_cols) > 0
    if add_datetime_featurizer and estimator_class.model_family not in [
        ModelFamily.ARIMA,
        ModelFamily.PROPHET,
    ]:
        components.append(DateTimeFeaturizer)
    return components


def _get_natural_language(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    text_columns = list(X.ww.select("NaturalLanguage", return_schema=True).columns)
    if len(text_columns) > 0:
        components.append(NaturalLanguageFeaturizer)
    return components


def _get_imputer(X, y, problem_type, estimator_class, sampler_name=None):
    components = []

    input_logical_types = {type(lt) for lt in X.ww.logical_types.values()}
    text_columns = list(X.ww.select("NaturalLanguage", return_schema=True).columns)

    types_imputer_handles = {
        logical_types.AgeNullable,
        logical_types.Boolean,
        logical_types.BooleanNullable,
        logical_types.Categorical,
        logical_types.Double,
        logical_types.Integer,
        logical_types.IntegerNullable,
        logical_types.URL,
        logical_types.EmailAddress,
        logical_types.Datetime,
    }

    if len(input_logical_types.intersection(types_imputer_handles)) or len(
        text_columns
    ):
        components.append(Imputer)

    return components


def _get_ohe(X, y, problem_type, estimator_class, sampler_name=None):
    components = []

    # The URL and EmailAddress Featurizers will create categorical columns
    categorical_cols = list(
        X.ww.select(
            ["category", "URL", "EmailAddress", "BooleanNullable"], return_schema=True
        ).columns
    )
    if len(categorical_cols) > 0 and estimator_class not in {
        CatBoostClassifier,
        CatBoostRegressor,
    }:
        components.append(OneHotEncoder)
    return components


def _get_sampler(X, y, problem_type, estimator_class, sampler_name=None):
    components = []

    sampler_components = {
        "Undersampler": Undersampler,
        "Oversampler": Oversampler,
    }
    if sampler_name is not None:
        try:
            import_or_raise(
                "imblearn.over_sampling", error_msg="imbalanced-learn is not installed"
            )
            components.append(sampler_components[sampler_name])
        except ImportError:
            logger.warning(
                "Could not import imblearn.over_sampling, so defaulting to use Undersampler"
            )
            components.append(Undersampler)
    return components


def _get_standard_scaler(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    if estimator_class and estimator_class.model_family == ModelFamily.LINEAR_MODEL:
        components.append(StandardScaler)
    return components


def _get_time_series_featurizer(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    if (
        is_time_series(problem_type)
        and estimator_class.model_family != ModelFamily.ARIMA
    ):
        components.append(TimeSeriesFeaturizer)
    return components


def _get_drop_nan_rows_transformer(
    X, y, problem_type, estimator_class, sampler_name=None
):
    components = []
    if is_time_series(problem_type) and estimator_unable_to_handle_nans(
        estimator_class
    ):
        components.append(DropNaNRowsTransformer)
    return components
