# from evalml.pipelines import get_pipelines_by_model_type
from sklearn.model_selection import StratifiedKFold

from .auto_base import AutoBase

from evalml.objectives import get_objective
from evalml.problem_types import ProblemTypes


class AutoClassifier(AutoBase):
    """Automatic pipeline search class for classification problems"""

    def __init__(self,
                 objective=None,
                 multiclass=False,
                 max_pipelines=None,
                 max_time=None,
                 model_types=None,
                 cv=None,
                 tuner=None,
                 detect_label_leakage=True,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
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

            model_types (list): The model types to search. By default searches over all
                model_types. Run evalml.list_model_types("classification") to see options.

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

            verbose (bool): If True, turn verbosity on. Defaults to True
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
            problem_type = self.set_problem_type(objective, multiclass)

        super().__init__(
            tuner=tuner,
            objective=objective,
            cv=cv,
            max_pipelines=max_pipelines,
            max_time=max_time,
            model_types=model_types,
            problem_type=problem_type,
            detect_label_leakage=detect_label_leakage,
            start_iteration_callback=start_iteration_callback,
            add_result_callback=add_result_callback,
            additional_objectives=additional_objectives,
            random_state=random_state,
            verbose=verbose
        )

    def set_problem_type(self, objective, multiclass):
        """Sets the problem type of the AutoClassifier to either binary or multiclass.

        If there is an objective either:
            a. Set problem_type to MULTICLASS if objective is only multiclass and multiclass is false
            b. Set problem_type to MUTLICLASS if multiclass is true
            c. Default to BINARY

        Arguments:
            objective (Object): the objective to optimize
            multiclass (bool): boolean representing whether search is for multiclass problems or not

        Returns:
            ProblemTypes enum representing type of problem to set AutoClassifier to

        """
        problem_type = ProblemTypes.BINARY
        # if exclusively multiclass: infer
        if [ProblemTypes.MULTICLASS] == get_objective(objective).problem_types:
            problem_type = ProblemTypes.MULTICLASS
        elif multiclass:
            problem_type = ProblemTypes.MULTICLASS
        return problem_type
