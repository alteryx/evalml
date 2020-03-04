from sklearn.model_selection import StratifiedKFold

from .auto_base import AutoBase

from evalml.objectives import ROC, ConfusionMatrix, get_objective
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
                 model_families=None,
                 cv=None,
                 tuner=None,
                 detect_label_leakage=True,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 n_jobs=-1,
                 verbose=True):
        """Automated classifier pipeline search

        Arguments:
            objective (Object): the objective to optimize

            multiclass (bool): If True, expecting multiclass data. By default: False.

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

            model_families (list): The model types to search. By default searches over all
                model_families. Run evalml.list_model_families("binary") to see options.

            cv: cross validation method to use. By default StratifiedKFold

            tuner: the tuner class to use. Defaults to scikit-optimize tuner

            detect_label_leakage (bool): If True, check input features for label leakage and
                warn if found. Defaults to true.

            start_iteration_callback (callable): function called before each pipeline training iteration.
                Passed two parameters: pipeline_class, parameters.

            add_result_callback (callable): function called after each pipeline training iteration.
                Passed two parameters: results, trained_pipeline.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_state (int): the random_state

            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

            verbose (boolean): If True, turn verbosity on. Defaults to True
        """

        if cv is None:
            cv = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)

        # set default objective if none provided
        if objective is None and not multiclass:
            objective = "precision"
            problem_type = ProblemTypes.BINARY
        elif objective is None and multiclass:
            objective = "precision_micro"
            problem_type = ProblemTypes.MULTICLASS
        else:
            problem_type = self._set_problem_type(objective, multiclass)

        super().__init__(
            tuner=tuner,
            objective=objective,
            cv=cv,
            max_pipelines=max_pipelines,
            max_time=max_time,
            patience=patience,
            tolerance=tolerance,
            model_families=model_families,
            problem_type=problem_type,
            detect_label_leakage=detect_label_leakage,
            start_iteration_callback=start_iteration_callback,
            add_result_callback=add_result_callback,
            additional_objectives=additional_objectives,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )

        # hacky, disallows non-numeric metrics from being primary objective
        if isinstance(self.objective, ConfusionMatrix) or isinstance(self.objective, ROC):
            raise RuntimeError("Cannot use Confusion Matrix or ROC as the main objective.")

        # if ROC and ConfusionMatrix not specified as additional objectives, add so we can calculate plots
        plot_metrics = [ROC(), ConfusionMatrix()]
        for metric in plot_metrics:
            if self.problem_type in metric.problem_types:
                existing_metric = next((obj for obj in self.additional_objectives if obj.name == metric.name), None)
                if existing_metric is None:
                    self.additional_objectives.append(get_objective(metric))

    def _set_problem_type(self, objective, multiclass):
        """Sets the problem type of the AutoClassificationSearch to either binary or multiclass.

        If there is an objective either:
            a. Set problem_type to MULTICLASS if objective is only multiclass and multiclass is false
            b. Set problem_type to MUTLICLASS if multiclass is true
            c. Default to BINARY

        Arguments:
            objective (Object): the objective to optimize
            multiclass (bool): boolean representing whether search is for multiclass problems or not

        Returns:
            ProblemTypes enum representing type of problem to set AutoClassificationSearch to

        """
        problem_type = ProblemTypes.BINARY
        # if exclusively multiclass: infer
        if [ProblemTypes.MULTICLASS] == get_objective(objective).problem_types:
            problem_type = ProblemTypes.MULTICLASS
        elif multiclass:
            problem_type = ProblemTypes.MULTICLASS
        return problem_type
