"""Component that imputes missing data according to a specified imputation strategy per column."""
import warnings

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.transformers.imputers.simple_imputer import (
    SimpleImputer,
)
from evalml.utils import infer_feature_types


class PerColumnImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy per column.

    Args:
        impute_strategies (dict): Column and {"impute_strategy": strategy, "fill_value":value} pairings.
            Valid values for impute strategy include "mean", "median", "most_frequent", "constant" for numerical data,
            and "most_frequent", "constant" for object data types. Defaults to None, which uses "most_frequent" for all columns.
            When impute_strategy == "constant", fill_value is used to replace missing data.
            When None, uses 0 when imputing numerical data and "missing_value" for strings or object data types.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Per Column Imputer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        impute_strategies=None,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "impute_strategies": impute_strategies,
        }
        self.imputers = None
        self.impute_strategies = impute_strategies or dict()
        if not isinstance(self.impute_strategies, dict):
            raise ValueError(
                "`impute_strategies` is not a dictionary. Please provide in Column and {`impute_strategy`: strategy, `fill_value`:value} pairs. ",
            )
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits imputers on input data.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features] to fit.
            y (pd.Series, optional): The target training data of length [n_samples]. Ignored.

        Returns:
            self
        """
        X = infer_feature_types(X)
        self.imputers = dict()

        columns_to_impute = self.impute_strategies.keys()
        if len(columns_to_impute) == 0:
            warnings.warn(
                "No columns to impute. Please check `impute_strategies` parameter.",
            )

        for column in columns_to_impute:
            strategy_dict = self.impute_strategies.get(column, dict())
            strategy = strategy_dict["impute_strategy"]
            fill_value = strategy_dict.get("fill_value", None)
            self.imputers[column] = SimpleImputer(
                impute_strategy=strategy,
                fill_value=fill_value,
            )

        for column, imputer in self.imputers.items():
            imputer.fit(X.ww[[column]])

        return self

    def transform(self, X, y=None):
        """Transforms input data by imputing missing values.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features] to transform.
            y (pd.Series, optional): The target training data of length [n_samples]. Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_ww = infer_feature_types(X)
        original_schema = X_ww.ww.schema

        cols_to_drop = []
        for column, imputer in self.imputers.items():
            transformed = imputer.transform(X_ww.ww[[column]])
            if transformed.empty:
                cols_to_drop.append(column)
            else:
                X_ww.ww[column] = transformed[column]
        X_t = X_ww.ww.drop(cols_to_drop)
        X_t.ww.init(schema=original_schema.get_subset_schema(X_t.columns))
        return X_t
