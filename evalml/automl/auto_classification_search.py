# from evalml.pipelines import get_pipelines_by_model_type
from sklearn.model_selection import StratifiedKFold

from .auto_base import AutoBase

from evalml.objectives import get_objective
from evalml.problem_types import ProblemTypes


class AutoClassificationSearch(AutoBase):
    """Automatic pipeline search class for classification problems"""

    def __init__(self,
                 objective=None,
                 multiclass=False,
                 max_pipelines=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 allowed_model_families=None,
                 cv=None,
                 tuner=None,
                 detect_label_leakage=True,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 n_jobs=-1,
                 optimize_thresholds=False):
        """Automated classifier pipeline search

        Arguments:
            objective (Object): The objective to optimize for.
                Defaults to LogLossBinary for binary classification problems and
                LogLossMulticlass for multiclass classification problems.

            multiclass (bool): If True, expecting multiclass data. Defaults to False.

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

            allowed_model_families (list): The model families to search. By default, searches over all
                model families. Run evalml.list_model_families("binary") to see options. Change `binary`
                to `multiclass` if your problem type is different.

            cv: cross-validation method to use. Defaults to StratifiedKFold.

            tuner: the tuner class to use. Defaults to scikit-optimize tuner

            detect_label_leakage (bool): If True, check input features for label leakage and
                warn if found. Defaults to true.

            start_iteration_callback (callable): function called before each pipeline training iteration.
                Passed two parameters: pipeline_class, parameters.

            add_result_callback (callable): function called after each pipeline training iteration.
                Passed two parameters: results, trained_pipeline.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

        """

        if cv is None:
            cv = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)

        # set default objective if none provided
        if objective is None and not multiclass:
            objective = "log_loss_binary"
            problem_type = ProblemTypes.BINARY
        elif objective is None and multiclass:
            objective = "log_loss_multi"
            problem_type = ProblemTypes.MULTICLASS
        else:
            objective = get_objective(objective)
            problem_type = objective.problem_type

        super().__init__(
            tuner=tuner,
            objective=objective,
            cv=cv,
            max_pipelines=max_pipelines,
            max_time=max_time,
            patience=patience,
            tolerance=tolerance,
            allowed_model_families=allowed_model_families,
            problem_type=problem_type,
            detect_label_leakage=detect_label_leakage,
            start_iteration_callback=start_iteration_callback,
            add_result_callback=add_result_callback,
            additional_objectives=additional_objectives,
            random_state=random_state,
            n_jobs=n_jobs,
            optimize_thresholds=optimize_thresholds
        )
