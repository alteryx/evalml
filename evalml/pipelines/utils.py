import numpy as np
import pandas as pd
from sklearn.inspection import \
    permutation_importance as sk_permutation_importance

from .binary_classification_pipeline import BinaryClassificationPipeline
from .classification import (
    CatBoostBinaryClassificationPipeline,
    CatBoostMulticlassClassificationPipeline,
    LogisticRegressionBinaryPipeline,
    LogisticRegressionMulticlassPipeline,
    RFBinaryClassificationPipeline,
    RFMulticlassClassificationPipeline,
    XGBoostBinaryPipeline,
    XGBoostMulticlassPipeline
)
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from .regression import (
    CatBoostRegressionPipeline,
    LinearRegressionPipeline,
    RFRegressionPipeline,
    XGBoostRegressionPipeline
)
from .regression_pipeline import RegressionPipeline

from evalml.exceptions import MissingComponentError
from evalml.model_family import handle_model_family
from evalml.pipelines.components import (
    CatBoostClassifier,
    CatBoostRegressor,
    DateTimeFeaturization,
    DropNullColumns,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    SimpleImputer,
    StandardScaler,
    XGBoostClassifier,
    XGBoostRegressor
)
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger

logger = get_logger(__file__)

_ALL_PIPELINES = [CatBoostBinaryClassificationPipeline,
                  CatBoostMulticlassClassificationPipeline,
                  LogisticRegressionBinaryPipeline,
                  LogisticRegressionMulticlassPipeline,
                  RFBinaryClassificationPipeline,
                  RFMulticlassClassificationPipeline,
                  XGBoostBinaryPipeline,
                  XGBoostMulticlassPipeline,
                  CatBoostRegressionPipeline,
                  LinearRegressionPipeline,
                  RFRegressionPipeline,
                  XGBoostRegressionPipeline]


_ALL_ESTIMATORS = [CatBoostClassifier,
                   CatBoostRegressor,
                   LinearRegressor,
                   LogisticRegressionClassifier,
                   RandomForestClassifier,
                   RandomForestRegressor,
                   XGBoostClassifier,
                   XGBoostRegressor]


def all_pipelines():
    """Returns a complete list of all supported pipeline classes.

    Returns:
        list[PipelineBase]: a list of pipeline classes
    """
    pipelines = []
    for pipeline_class in _ALL_PIPELINES:
        try:
            pipeline_class({})
            pipelines.append(pipeline_class)
        except (MissingComponentError, ImportError):
            pipeline_name = pipeline_class.name
            logger.debug('Pipeline {} failed import, withholding from all_pipelines'.format(pipeline_name))
    return pipelines


def get_pipelines(problem_type, model_families=None):
    """Returns the pipelines allowed for a particular problem type.

    Can also optionally filter by a list of model types.

    Arguments:
        problem_type (ProblemTypes or str): problem type to filter for
        model_families (list[ModelFamily] or list[str]): model families to filter for

    Returns:
        list[PipelineBase]: a list of pipeline classes
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")

    if model_families:
        model_families = [handle_model_family(model_family) for model_family in model_families]

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in all_pipelines():
        if problem_type == handle_problem_types(p.problem_type):
            problem_pipelines.append(p)

    if model_families is None:
        return problem_pipelines

    all_model_families = list_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

    pipelines = []

    for p in problem_pipelines:
        if p.model_family in model_families:
            pipelines.append(p)

    return pipelines


def list_model_families(problem_type):
    """List model type for a particular problem type

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        list[ModelFamily]: a list of model families
    """

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in all_pipelines():
        if problem_type == handle_problem_types(p.problem_type):
            problem_pipelines.append(p)

    return list(set([p.model_family for p in problem_pipelines]))


def all_estimators():
    """Returns a complete list of all supported estimator classes.

    Returns:
        list[Estimator]: a list of estimator classes
    """
    estimators = []
    for estimator_class in _ALL_ESTIMATORS:
        try:
            estimator_class()
            estimators.append(estimator_class)
        except (MissingComponentError, ImportError):
            estimator_name = estimator_class.name
            logger.debug('Estimator {} failed import, withholding from all_estimators'.format(estimator_name))
    return estimators


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
        model_families = list_model_families(problem_type)

    model_families = [handle_model_family(model_family) for model_family in model_families]
    all_model_families = list_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

    estimator_classes = []
    for estimator_class in all_estimators():
        if problem_type not in [handle_problem_types(supported_pt) for supported_pt in estimator_class.supported_problem_types]:
            continue
        if estimator_class.model_family not in model_families:
            continue
        estimator_classes.append(estimator_class)
    return estimator_classes


def _get_preprocessing_components(X, y, problem_type, estimator_class):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Arguments:
        X (pd.DataFrame): the input data of shape [n_samples, n_features]
        y (pd.Series): the target labels of length [n_samples]
        problem_type (ProblemTypes or str): problem type
        estimator_class (class):A class which subclasses Estimator estimator for pipeline

    Returns:
        list[Transformer]: a list of applicable preprocessing components to use with the estimator
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    pp_components = []
    all_null_cols = X.columns[X.isnull().all()]
    if len(all_null_cols) > 0:
        pp_components.append(DropNullColumns)
    X = X.drop(all_null_cols, axis=1)
    pp_components.append(SimpleImputer)

    datetime_cols = X.select_dtypes(include=[np.datetime64])
    add_datetime_featurization = len(datetime_cols.columns) > 0
    if add_datetime_featurization:
        pp_components.append(DateTimeFeaturization)

    # DateTimeFeaturization can create categorical columns
    categorical_cols = X.select_dtypes(include=['category', 'object'])
    if (add_datetime_featurization or len(categorical_cols.columns) > 0) and estimator_class not in {CatBoostClassifier, CatBoostRegressor}:
        pp_components.append(OneHotEncoder)

    if estimator_class in {LinearRegressor, LogisticRegressionClassifier}:
        pp_components.append(StandardScaler)
    return pp_components


def make_pipeline(X, y, estimator, problem_type):
    """Given input data, target data, an estimator class and the problem type,
        generates a pipeline class with a preprocessing chain which was recommended based on the inputs.
        The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type.

   Arguments:
        X (pd.DataFrame): the input data of shape [n_samples, n_features]
        y (pd.Series): the target labels of length [n_samples]
        estimator (Estimator): estimator for pipeline
        problem_type (ProblemTypes or str): problem type for pipeline to generate

    Returns:
        class: PipelineBase subclass with dynamically generated preprocessing components and specified estimator

    """
    problem_type = handle_problem_types(problem_type)
    if estimator not in get_estimators(problem_type):
        raise ValueError(f"{estimator.name} is not a valid estimator for problem type")
    preprocessing_components = _get_preprocessing_components(X, y, problem_type, estimator)
    complete_component_graph = preprocessing_components + [estimator]

    def get_pipeline_base_class(problem_type):
        """Returns pipeline base class for problem_type"""
        if problem_type == ProblemTypes.BINARY:
            return BinaryClassificationPipeline
        elif problem_type == ProblemTypes.MULTICLASS:
            return MulticlassClassificationPipeline
        elif problem_type == ProblemTypes.REGRESSION:
            return RegressionPipeline

    base_class = get_pipeline_base_class(problem_type)

    class GeneratedPipeline(base_class):
        component_graph = complete_component_graph
    return GeneratedPipeline


def get_permutation_importances(pipeline, X, y, n_repeats=5, n_jobs=None, random_state=0):
    """Calculates permutation importance for features.

    Arguments:
        pipeline (PipelineBase or subclass): fitted pipeline
        X (pd.DataFrame): the input data used to score and compute permutation importance
        y (pd.Series): the target labels
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

    Returns:
        Mean feature importance scores over 5 shuffles.
    """
    def scorer(pipeline, X, y):
        scores = pipeline.score(X, y, objectives=["RMSE"])
        return -scores['Root Mean Squared Error']
    perm_importances = sk_permutation_importance(pipeline, X, y, n_repeats=n_repeats, scoring=scorer, n_jobs=n_jobs, random_state=random_state)
    mean_perm_importances = perm_importances["importances_mean"]
    if not isinstance(X, pd.DataFrame):
        return pd.DataFrame(mean_perm_importances)
    feature_names = list(X.columns)
    mean_perm_importances = list(zip(feature_names, mean_perm_importances))
    mean_perm_importances.sort(key=lambda x: -abs(x[1]))
    return pd.DataFrame(mean_perm_importances, columns=["feature", "importance"])


# def graph_permutation_importances(pipeline, X, y, show_all_features=True):
#     """Generate a bar graph of the pipeline's feature permutation importances

#         Arguments:
#             show_all_features (bool, optional) : If true, graph features with an importance value of zero. Defaults to False.
#         Returns:
#             plotly.Figure, a bar graph showing features and their importances
#         """
#         go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
#         get_permutation_importances(pipeline, X, y)
#         feat_imp['importance'] = abs(feat_imp['importance'])

#         if not show_all_features:
#             # Remove features with zero importance
#             feat_imp = feat_imp[feat_imp['importance'] != 0]

#         # List is reversed to go from ascending order to descending order
#         feat_imp = feat_imp.iloc[::-1]

#         title = 'Feature Importances'
#         subtitle = 'May display fewer features due to feature selection'
#         data = [go.Bar(
#             x=feat_imp['importance'],
#             y=feat_imp['feature'],
#             orientation='h'
#         )]

#         layout = {
#             'title': '{0}<br><sub>{1}</sub>'.format(title, subtitle),
#             'height': 800,
#             'xaxis_title': 'Feature Importance',
#             'yaxis_title': 'Feature',
#             'yaxis': {
#                 'type': 'category'
#             }
#         }

#         fig = go.Figure(data=data, layout=layout)
#         return fig
