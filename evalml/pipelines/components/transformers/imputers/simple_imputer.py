import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components.transformers import Transformer


class SimpleImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy"""
    name = 'Simple Imputer'
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}

    def __init__(self, impute_strategy="most_frequent", fill_value=None, random_state=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
               numerical data, and "most_frequent", "constant" for object data types.
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
        """
        parameters = {"impute_strategy": impute_strategy,
                      "fill_value": fill_value}
        parameters.update(kwargs)
        imputer = SkImputer(strategy=impute_strategy,
                            fill_value=fill_value,
                            **kwargs)
        self._all_null_cols = None
        super().__init__(parameters=parameters,
                         component_obj=imputer,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Fits imputer to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._component_obj.fit(X, y)
        self._all_null_cols = set(X.columns) - set(X.dropna(axis=1, how='all').columns)
        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        if self._all_null_cols is None:
            raise RuntimeError("Must fit transformer before calling transform!")
        X_t = self._component_obj.transform(X)
        if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
            # skLearn's SimpleImputer loses track of column type, so we need to restore
            X_null_dropped = X.drop(self._all_null_cols, axis=1)
            if X_null_dropped.empty:
                return pd.DataFrame(X_t, columns=X_null_dropped.columns)
            X_t = pd.DataFrame(X_t, columns=X_null_dropped.columns).astype(X_null_dropped.dtypes.to_dict())
        return X_t

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd. DataFrame): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """
        return self.fit(X, y).transform(X, y)
