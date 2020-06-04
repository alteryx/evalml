from evalml.pipelines.components.transformers import Transformer


class DateTimeFeaturization(Transformer):
    """"""
    name = "DateTime Featurization Component"
    hyperparameter_ranges = {}

    def __init__(self, features_to_extract=None, random_state=0):
        """Extracts features from DateTime columns

        Arguments:
            features_to_extract (list)
        """
        if features_to_extract is None:
            features_to_extract = ["year", "month", "day_of_week", "hour"]
        parameters = {"features_to_extract": features_to_extract}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing DateTime columns, and then dropping those DateTime columns

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        pass
