from sklearn.model_selection import KFold

from .auto_search_base import AutoSearchBase

from evalml.problem_types import ProblemTypes


class AutoRegressionSearch(AutoSearchBase):
    """Automatic pipeline search for regression problems

    """

    def __init__(self,
                 objective=None,
                 max_pipelines=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 allowed_pipelines=None,
                 allowed_model_families=None,
                 cv=None,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 n_jobs=-1,
                 tuner_class=None,
                 verbose=True):
        """Automated regressors pipeline search

        Arguments:
            objective (Object): The objective to optimize for.
                Defaults to R2.

            max_pipelines (int): Maximum number of pipelines to search. If max_pipelines and
                max_time is not set, then max_pipelines will default to max_pipelines of 5.

            max_time (int, str): Maximum time to search for pipelines.
                This will not start a new pipeline search after the duration
                has elapsed. If it is an integer, then the time will be in seconds.
                For strings, time can be specified as seconds, minutes, or hours.

            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search.
                The default of None indicates all pipelines for this problem type are allowed.

            allowed_model_families (list(str, ModelFamily)): The model families to search. The default of None searches over all
                model families. Run evalml.list_model_families("regression") to see options.

            patience (int): Number of iterations without improvement to stop search early. Must be positive.
                If None, early stopping is disabled. Defaults to None.

            tolerance (float): Minimum percentage difference to qualify as score improvement for early stopping.
                Only applicable if patience is not None. Defaults to None.

            cv: cross validation method to use. By default StratifiedKFold

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
        if objective is None:
            objective = "R2"

        problem_type = ProblemTypes.REGRESSION

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
            verbose=verbose
        )
