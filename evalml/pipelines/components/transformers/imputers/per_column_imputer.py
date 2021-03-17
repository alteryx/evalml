from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.transformers.imputers.simple_imputer import (
    SimpleImputer
)
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class PerColumnImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy per column"""
    name = 'Per Column Imputer'
    hyperparameter_ranges = {}

    def __init__(self, impute_strategies=None, default_impute_strategy="most_frequent", random_seed=0, **kwargs):
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

            random_seed (int): Seed for the random number generator. Defaults to 0.
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
                         random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits imputers on input data

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features] to fit.
            y (ww.DataColumn, pd.Series, optional): The target training data of length [n_samples]. Ignored.

        Returns:
            self
        """
        X = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
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
        """Transforms input data by imputing missing values.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features] to transform.
            y (ww.DataColumn, pd.Series, optional): The target training data of length [n_samples]. Ignored.

        Returns:
            ww.DataTable: Transformed X
        """
        X_ww = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        X_t = X.copy()
        cols_to_drop = []
        for column, imputer in self.imputers.items():
            transformed = imputer.transform(X[[column]]).to_dataframe()
            if transformed.empty:
                cols_to_drop.append(column)
            else:
                X_t[column] = transformed[column]
        X_t = X_t.drop(cols_to_drop, axis=1)
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_t)
