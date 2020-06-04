from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components.transformers import Transformer


class PerColumnImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy per column"""
    name = 'Per Column Imputer'
    hyperparameter_ranges = {}

    def __init__(self, impute_strategies=None, default_impute_strategy="most_frequent", random_state=0):
        """Initializes a transformer that imputes missing data according to the specified imputation strategy per column."

        Arguments:
            impute_strategies (dict): Column and either impute_strategy or (impute_strategy, fill_value) pairings.
                Dict values can be either a singular string strategy or a tuple with a strategy and fill value.

                Valid values for impute strategy include "mean", "median", "most_frequent", "constant" for numerical data,
                and "most_frequent", "constant" for object data types. Defaults to "most_frequent" for all columns.

                When impute_strategy == "constant", fill_value is used to replace missing data.
                Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
            
            default_impute_strategy (str): Impute strategy to fall back on when none is provided for a certain column.
                Valid values include "mean", "median", "most_frequent", "constant" for numerical data,
                and "most_frequent", "constant" for object data types. Defaults to "most_frequent"
        """
        self.imputers = None
        self.default_impute_strategy = default_impute_strategy

        self.impute_strategies = impute_strategies or dict()
        parameters = {"impute_strategies": impute_strategies}

        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _init_imputers(self, X):
        imputers = dict()
        for column in X.columns:
            strategy = self.impute_strategies.get(column, self.default_impute_strategy)
            print(strategy)
            if isinstance(strategy, tuple):
                imputers[column] = SkImputer(strategy=strategy[0], fill_value=strategy[1])
            else:
                imputers[column] = SkImputer(strategy=strategy)
            print(imputers[column])
        return imputers

    def fit(self, X, y=None):
        """Fits imputers on data X

        Arguments:
            X (pd.DataFrame): Data to fit
            y (pd.Series, optional): Input Labels
        Returns:
            self
        """
        self.imputers = self._init_imputers(X)
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
        for column, imputer in self.imputers.items():
            X_t[column] = imputer.transform(X[[column]]).astype(X.dtypes[column])
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
