from sklearn.model_selection import KFold

from .auto_search_base import AutoSearchBase

from evalml.objectives import get_objective, standard_metrics
from evalml.problem_types import ProblemTypes


class ObjectiveProblemTypeError(Exception):
    """An exception thrown when a given objective and problem_type do not match"""
    pass


class AutoMLSearch(AutoSearchBase):
    """Automatic pipeline search class"""

    defaults = {'regression': {'type': ProblemTypes.REGRESSION, 'objective': standard_metrics.R2()},
                'binary': {'type': ProblemTypes.BINARY, 'objective': standard_metrics.LogLossBinary()},
                'multiclass': {'type': ProblemTypes.MULTICLASS, 'objective': standard_metrics.LogLossMulticlass()}}

    def __init__(self,
                 problem_type=None,
                 objective='auto',
                 max_pipelines=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 cv=None,
                 allowed_pipelines=None,
                 allowed_model_families=None,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 n_jobs=-1,
                 tuner_class=None,
                 verbose=True,
                 optimize_thresholds=False):
        """Automated pipeline search

        Arguments:
            problem_type (str): Choice of 'regression', 'binary', or 'multiclass', depending on the desired kind of
                problem.

            objective (Object): The objective to optimize for. When set to auto, chooses:
                LogLossBinary for binary classification problems,
                LogLossMulticlass for multiclass classification problems, and
                R2 for regression problems.

            max_pipelines (int): Maximum number of pipelines to search. If max_pipelines and
                max_time is not set, then max_pipelines will default to max_pipelines of 5.

            max_time (int, str): Maximum time to search for pipelines.
                This will not start a new pipeline search after the duration
                has elapsed. If it is an integer, then the time will be in seconds.
                For strings, time can be specified as seconds, minutes, or hours.

            patience (int): Number of iterations without improvement to stop search early. Must be positive.
                If None, early stopping is disabled. Defaults to None.

            tolerance (float): Minimum percentage difference to qualify as score improvement for early stopping.
                Only applicable if patience is not None. Defaults to None.

            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search.
                The default of None indicates all pipelines for this problem type are allowed. Setting this field will cause
                allowed_model_families to be ignored.

            allowed_model_families (list(str, ModelFamily)): The model families to search. The default of None searches over all
                model families. Run evalml.list_model_families("binary") to see options. Change `binary`
                to `multiclass` or `regression` depending on the problem type. Note that if allowed_pipelines is provided,
                this parameter will be ignored.

            cv: cross-validation method to use. Defaults to StratifiedKFold.

            tuner_class: the tuner class to use. Defaults to scikit-optimize tuner

            start_iteration_callback (callable): function called before each pipeline training iteration.
                Passed two parameters: pipeline_class, parameters.

            add_result_callback (callable): function called after each pipeline training iteration.
                Passed two parameters: results, trained_pipeline.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

            verbose (boolean): If True, turn verbosity on. Defaults to True
        """
        if problem_type is None:
            raise ObjectiveProblemTypeError('choose one of (binary, multiclass, regression) as problem_type')

        if objective == 'auto':
            objective = self.defaults[problem_type]['objective']
        else:
            objective = get_objective(objective)

        problem_type = self.defaults[problem_type]['type']

        if not objective.problem_type == problem_type:
            raise ObjectiveProblemTypeError('objective does not match given problem type')

        if cv is None:
            cv = KFold(n_splits=3, random_state=random_state)

        super().__init__(
            problem_type=problem_type,
            objective=objective,
            max_pipelines=max_pipelines,
            max_time=max_time,
            patience=patience,
            tolerance=tolerance,
            cv=cv,
            allowed_pipelines=allowed_pipelines,
            allowed_model_families=allowed_model_families,
            start_iteration_callback=start_iteration_callback,
            add_result_callback=add_result_callback,
            additional_objectives=additional_objectives,
            random_state=random_state,
            n_jobs=n_jobs,
            tuner_class=tuner_class,
            verbose=verbose,
            optimize_thresholds=optimize_thresholds
        )
