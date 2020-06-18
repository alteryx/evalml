from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.transformers.imputers.simple_imputer import (
    SimpleImputer
)


class PerColumnImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy per column"""
    name = 'Per Column Imputer'
    hyperparameter_ranges = {}

    def __init__(self, impute_strategies=None, default_impute_strategy="most_frequent", random_state=0, **kwargs):
        """Initializes a transformer that imputes missing data according to the specified imputation strategy per column."

        Arguments:
            impute_strategies (dict): Column and {"impute_strategy": strategy, "fill_value":value} pairings.
                Valid values for impute strategy include "mean", "median", "most_frequent", "constant" for numerical data,
                and "most_frequent", "constant" for object data types. Defaults to "most_frequent" for all columns.

                When impute_strategy == "constant", fill_value is used to replace missing data.
                Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.

            default_impute_strategy (str): Impute strategy to fall back on when none is provided for a certain column.
                Valid values include "mean", "median", "most_frequent", "constant" for numerical data,
                and "most_frequent", "constant" for object data types. Defaults to "most_frequent"
        """
        parameters = {"impute_strategies": impute_strategies,
                      "default_impute_strategy": default_impute_strategy}
        self.imputers = None
        self.default_impute_strategy = default_impute_strategy
        self.impute_strategies = impute_strategies or dict()

        if not isinstance(self.impute_strategies, dict):
            raise ValueError("`impute_strategies` is not a dictionary. Please provide in Column and {`impute_strategy`: strategy, `fill_value`:value} pairs. ")

        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Fits imputers on data X

        Arguments:
            X (pd.DataFrame): Data to fit
            y (pd.Series, optional): Input Labels
        Returns:
            self
        """
        self.imputers = dict()
        for column in X.columns:
            strategy_dict = self.impute_strategies.get(column, dict())
            strategy = strategy_dict.get('impute_strategy', self.default_impute_strategy)
            fill_value = strategy_dict.get('fill_value', None)
            self.imputers[column] = SimpleImputer(impute_strategy=strategy, fill_value=fill_value)

        for column, imputer in self.imputers.items():
            imputer.fit(X[[column]])

        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = X.copy()
        cols_to_drop = []
        for column, imputer in self.imputers.items():
            transformed = imputer.transform(X[[column]])
            if transformed.empty:
                cols_to_drop.append(column)
            else:
                X_t[column] = transformed
        X_t = X_t.drop(cols_to_drop, axis=1)
        return X_t

    def fit_transform(self, X, y=None):
        """Fits imputer on data X then imputes missing values in X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """

        self.fit(X, y)
        return self.transform(X, y)
