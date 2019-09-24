# from evalml.pipelines import get_pipelines_by_model_type
from sklearn.model_selection import StratifiedKFold

from .auto_base import AutoBase

from evalml.objectives import get_objective
from evalml.problem_types import ProblemTypes


class AutoClassifier(AutoBase):
    """Automatic pipeline search for classification problems"""

    def __init__(self,
                 objective=None,
                 multiclass=False,
                 max_pipelines=5,
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

            max_pipelines (int): maximum number of pipelines to search

            max_time (int): maximum time in seconds to search for pipelines.
                won't start new pipeline search after this duration has elapsed

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

            verbose (boolean): If True, turn verbosity on. Defaults to True
        """

        if cv is None:
            cv = StratifiedKFold(n_splits=3, random_state=random_state)

        problem_type = ProblemTypes.BINARY

        """
        If there is an objective either:
            a. Set problem_type to MULTICLASS if objective is only multiclass and multiclass was false
            b. Check if objective and multiclass is compatible
            c. Set problem_type to MUTLiCLASS
        """
        if objective is not None:
            if multiclass is False:
                if [ProblemTypes.MULTICLASS] == get_objective(objective).problem_types:
                    problem_type = ProblemTypes.MULTICLASS
            elif multiclass and ProblemTypes.MULTICLASS not in get_objective(objective).problem_types:
                raise ValueError("Multiclass is set to true and provided objective is not a multiclass objective")
            elif multiclass:
                problem_type = ProblemTypes.MULTICLASS

        # if there is no provided objective set to precision or precision_micro if multiclass was set to True.
        if objective is None and not multiclass:
            objective = "precision"
        elif objective is None and multiclass:
            objective = "precision_micro"
            problem_type = ProblemTypes.MULTICLASS

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
            random_state=random_state,
            verbose=verbose,
            additional_objectives=additional_objectives
        )
