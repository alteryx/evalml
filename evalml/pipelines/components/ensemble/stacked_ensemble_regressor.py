"""Stacked Ensemble Regressor."""
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ElasticNetRegressor
from evalml.pipelines.components.ensemble import StackedEnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleRegressor(StackedEnsembleBase):
    """Stacked Ensemble Regressor.

    Arguments:
        final_estimator (Estimator or subclass): The regressor used to combine the base estimators. If None, uses ElasticNetRegressor.
        n_jobs (int or None): Integer describing level of parallelism used for pipelines. None and 1 are equivalent.
            If set to -1, all CPUs are used. For n_jobs greater than -1, (n_cpus + 1 + n_jobs) are used. Defaults to -1.
            - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> from evalml.pipelines.component_graph import ComponentGraph
        >>> from evalml.pipelines.components.estimators.regressors.rf_regressor import RandomForestRegressor
        >>> from evalml.pipelines.components.estimators.regressors.elasticnet_regressor import ElasticNetRegressor
        ...
        >>> component_graph = {
        ...     "Random Forest": [RandomForestRegressor(random_seed=3), "X", "y"],
        ...     "Random Forest B": [RandomForestRegressor(random_seed=4), "X", "y"],
        ...     "Stacked Ensemble": [
        ...         StackedEnsembleRegressor(n_jobs=1, final_estimator=RandomForestRegressor()),
        ...         "Random Forest.x",
        ...         "Random Forest B.x",
        ...         "y",
        ...     ],
        ... }
        ...
        >>> cg = ComponentGraph(component_graph)
        >>> assert cg.default_parameters == {
        ...     'Random Forest Regressor': {'n_estimators': 100,
        ...                                 'max_depth': 6,
        ...                                 'n_jobs': -1},
        ...     'Stacked Ensemble Regressor': {'final_estimator': ElasticNetRegressor,
        ...                                    'n_jobs': -1}}
    """

    name = "Stacked Ensemble Regressor"
    model_family = ModelFamily.ENSEMBLE
    """ModelFamily.ENSEMBLE"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""
    hyperparameter_ranges = {}
    """{}"""
    _default_final_estimator = ElasticNetRegressor
