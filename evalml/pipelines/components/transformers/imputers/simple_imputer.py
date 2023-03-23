"""Component that imputes missing data according to a specified imputation strategy."""
import pandas as pd
import woodwork
from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types
from evalml.utils.nullable_type_utils import _get_new_logical_types_for_imputed_data


class SimpleImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy.  Natural language columns are ignored.

    Args:
        impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
           numerical data, and "most_frequent", "constant" for object data types.
        fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
           Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    """

    name = "Simple Imputer"
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}
    """{
        "impute_strategy": ["mean", "median", "most_frequent"]
    }"""

    def __init__(
        self, impute_strategy="most_frequent", fill_value=None, random_seed=0, **kwargs
    ):
        parameters = {"impute_strategy": impute_strategy, "fill_value": fill_value}
        parameters.update(kwargs)
        self.impute_strategy = impute_strategy
        imputer = SkImputer(
            strategy=impute_strategy,
            fill_value=fill_value,
            missing_values=pd.NA,
            **kwargs,
        )
        self._all_null_cols = None
        super().__init__(
            parameters=parameters,
            component_obj=imputer,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits imputer to data. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self

        Raises:
            ValueError: if the SimpleImputer receives a dataframe with both Boolean and Categorical data.

        """
        X = infer_feature_types(X)

        if set([lt.type_string for lt in X.ww.logical_types.values()]) == {
            "boolean",
            "categorical",
        }:
            raise ValueError(
                "SimpleImputer cannot handle dataframes with both boolean and categorical features.  Use Imputer instead.",
            )

        nan_ratio = X.isna().sum() / X.shape[0]

        # Keep track of the different types of data in X
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()
        self._natural_language_cols = list(
            X.ww.select(
                "NaturalLanguage",
                return_schema=True,
            ).columns.keys(),
        )

        # Only impute data that is not natural language columns or fully null
        self._cols_to_impute = [
            col
            for col in X.columns
            if col not in self._natural_language_cols and col not in self._all_null_cols
        ]

        # If there are no columns to impute, return early
        if not self._cols_to_impute:
            return self

        X = X[self._cols_to_impute]
        if (X.dtypes == bool).all():
            # Ensure that _component_obj still gets fit so that if any of the dtypes are different
            # at transform, we've fit the component. This is needed because sklearn doesn't allow
            # data with only bool dtype to be passed in.
            X = X.astype("boolean")

        self._component_obj.fit(X, y)
        return self

    def transform(self, X, y=None):
        """Transforms input by imputing missing values. 'None' and np.nan values are treated as the same.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        # Record original data
        X = infer_feature_types(X)
        original_schema = X.ww.schema
        original_index = X.index

        # separate out just the columns we are imputing
        X_t = X[self._cols_to_impute]
        if not self._cols_to_impute or (X_t.dtypes == bool).all():
            # If there are no columns to impute or all columns to impute are bool dtype,
            # which will never have null values, return the original data without any fully null columns
            not_all_null_cols = [
                col for col in X.columns if col not in self._all_null_cols
            ]
            return X.ww[not_all_null_cols]

        # Transform the data
        X_t = self._component_obj.transform(X_t)
        X_t = pd.DataFrame(X_t, columns=self._cols_to_impute)

        # Reinit woodwork, maintaining original types where possible
        imputed_schema = original_schema.get_subset_schema(self._cols_to_impute)
        new_logical_types = _get_new_logical_types_for_imputed_data(
            impute_strategy=self.impute_strategy,
            original_schema=imputed_schema,
        )
        X_t.ww.init(schema=imputed_schema, logical_types=new_logical_types)

        # Add back in the unchanged original natural language columns that we want to keep
        if len(self._natural_language_cols) > 0:
            X_t = woodwork.concat_columns([X_t, X.ww[self._natural_language_cols]])
            # reorder columns to match original
            X_t = X_t.ww[[col for col in original_schema.columns if col in X_t.columns]]

        if self._cols_to_impute:
            X_t.index = original_index
        return X_t

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X.

        Args:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series, optional): Target data.

        Returns:
            pd.DataFrame: Transformed X
        """
        return self.fit(X, y).transform(X, y)
