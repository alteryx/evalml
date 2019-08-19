# from evalml.pipelines import get_pipelines_by_model_type
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .auto_base import AutoBase

from evalml.objectives import standard_metrics


class AutoClassifier(AutoBase):
    """Automatic pipeline search for classification problems"""

    def __init__(self,
                 objective=None,
                 max_pipelines=5,
                 max_time=None,
                 model_types=None,
                 cv=None,
                 tuner=None,
                 random_state=0,
                 verbose=True):
        """Automated classifier pipeline search

        Arguments:
            objective (Object): the objective to optimize
            max_pipelines (int): maximum number of pipelines to search
            max_time (int): maximum time in seconds to search for pipelines.
                won't start new pipeline search after this duration has elapsed
            model_types (list): The model types to search. By default searches over all
                model_types. Run evalml.list_model_types("classification") to see options.
            cv: cross validation method to use. By default StratifiedKFold
            tuner: the tuner class to use. Defaults to scikit-optimize tuner
            random_state (int): the random_state
            verbose (boolean): If True, turn verbosity on. Defaults to True

        """
        if objective is None:
            objective = "precision"

        if cv is None:
            cv = StratifiedKFold(n_splits=3, random_state=random_state)

        default_objectives = [
            standard_metrics.F1(),
            standard_metrics.Precision(),
            standard_metrics.Recall(),
            standard_metrics.AUC(),
            standard_metrics.LogLoss()
        ]

        problem_type = "classification"

        super().__init__(
            tuner=tuner,
            objective=objective,
            cv=cv,
            max_pipelines=max_pipelines,
            max_time=max_time,
            model_types=model_types,
            problem_type=problem_type,
            default_objectives=default_objectives,
            random_state=random_state,
            verbose=verbose,
        )


if __name__ == "__main__":
    from evalml.objectives import FraudDetection
    from evalml.preprocessing import load_data
    from evalml.preprocessing import split_data

    filepath = "/Users/kanter/Documents/lead_scoring_app/fraud_demo/data/transactions.csv"
    X, y = load_data(filepath, index="id", label="fraud")

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    from sklearn.datasets import load_digits

    digits = load_digits()

    X_train, X_test, y_train, y_test = split_data(pd.DataFrame(digits.data), pd.Series(digits.target), test_size=.2, random_state=0)
    print(X_train)
    objective = FraudDetection(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    clf = AutoClassifier(objective="precision",
                         max_pipelines=3,
                         random_state=0)

    clf.fit(X_train, y_train)

    print(clf.rankings)

    print(clf.best_pipeline)
    print(clf.best_pipeline.score(X_test, y_test))

    clf.rankings.to_csv("rankings.csv")
