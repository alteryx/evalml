class AutoClassifier:

    def __init__(self, max_models=10, max_time=60*5, model_types=None):
        """Automated classifer search

        Arguments:
            max_models (int): maximum number of models to search
            max_time (int): maximum time in seconds to search for models
            model_types (list): The model types to search. By default searches over
                model_types

        """
        pass

    def fit(self, X, y, metric=None, feature_types=None):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]
            metric (Metric): the metric to optimize

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

        Returns:

            self
        """
        pass

    @property
    def rankings(self):
        """Returns the rankings of the models searched"""
        pass


    @property
    def best_model(self):
        """Returns the best model found"""
        pass


