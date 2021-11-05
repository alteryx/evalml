"""Stacked Ensemble Classifier."""
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ElasticNetClassifier
from evalml.pipelines.components.ensemble import StackedEnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleClassifier(StackedEnsembleBase):
    """Stacked Ensemble Classifier.

    Arguments:
        final_estimator (Estimator or subclass): The classifier used to combine the base estimators. If None, uses ElasticNetClassifier.
        n_jobs (int or None): Integer describing level of parallelism used for pipelines. None and 1 are equivalent.
            If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Defaults to -1.
            - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> from evalml.pipelines.component_graph import ComponentGraph
        >>> from evalml.pipelines.components.estimators.classifiers.decision_tree_classifier import DecisionTreeClassifier
        >>> from evalml.pipelines.components.estimators.classifiers.elasticnet_classifier import ElasticNetClassifier
        ...
        >>> component_graph = {
        ...     "Decision Tree": [DecisionTreeClassifier(random_seed=3), "X", "y"],
        ...     "Decision Tree B": [DecisionTreeClassifier(random_seed=4), "X", "y"],
        ...     "Stacked Ensemble": [
        ...         StackedEnsembleClassifier(n_jobs=1, final_estimator=DecisionTreeClassifier()),
        ...         "Decision Tree.x",
        ...         "Decision Tree B.x",
        ...         "y",
        ...     ],
        ... }
        ...
        >>> cg = ComponentGraph(component_graph)
        >>> assert cg.default_parameters == {
        ...     'Decision Tree Classifier': {'criterion': 'gini',
        ...                                  'max_features': 'auto',
        ...                                  'max_depth': 6,
        ...                                  'min_samples_split': 2,
        ...                                  'min_weight_fraction_leaf': 0.0},
        ...     'Stacked Ensemble Classifier': {'final_estimator': ElasticNetClassifier,
        ...                                     'n_jobs': -1}}
    """

    name = "Stacked Ensemble Classifier"
    model_family = ModelFamily.ENSEMBLE
    """ModelFamily.ENSEMBLE"""
    supported_problem_types = [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""
    hyperparameter_ranges = {}
    """{}"""
    _default_final_estimator = ElasticNetClassifier
