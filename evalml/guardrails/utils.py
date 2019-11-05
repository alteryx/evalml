import pandas as pd


def detect_label_leakage(X, y, threshold=.95):
    """Check if any of the features are highly correlated with the target.

    Currently only supports binary and numeric targets and features

    Args:
        X (pd.DataFrame): The input features to check
        y (pd.Series): the labels
        threshold (float): the correlation threshold to be considered leakage. Defaults to .95

    Returns:
        leakage, dictionary of features with leakage and corresponding threshold
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    corrs = X.corrwith(y).abs()
    out = corrs[corrs >= threshold]
    return out.to_dict()


def detect_highly_null(X, percent_threshold=.95):
    """ Checks if there are any highly-null columns in a dataframe.

    Args:
        X (DataFrame) : features
        percent_threshold(float): Require that percentage of null values to be considered "highly-null", defaults to .95

    Returns:
        A dictionary of features with column name or index and their percentage of null values
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    percent_null = (X.isnull().mean()).to_dict()
    highly_null_cols = {key: value for key, value in percent_null.items() if value >= percent_threshold}
    return highly_null_cols


def detect_id_columns(X, threshold=1.0):
    """Check if any of the features are ID columns.
    Currently performs these simple checks:
        - column name is "id"
        - column name ends in "_id"
        - column contains all unique values (and is not float / boolean)
    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the probability threshold to be considered an ID column. Defaults to 1.0
    Returns:
        A dictionary of features with column name or index and their probability of being ID columns
    """
    col_names = [str(col) for col in X.columns.tolist()]
    cols_named_id = [col for col in col_names if (col.lower() == "id")]  # columns whose name is "id"
    id_cols = {col: 0.95 for col in cols_named_id}

    non_id_types = ['float16', 'float32', 'float64', 'bool']
    X = X.select_dtypes(exclude=non_id_types)
    check_all_unique = (X.nunique() == len(X))
    cols_with_all_unique = check_all_unique[check_all_unique].index.tolist()  # columns whose values are all unique
    id_cols.update([(str(col), 1.0) if col in id_cols else (str(col), 0.95) for col in cols_with_all_unique])

    col_ends_with_id = [col for col in col_names if str(col).lower().endswith("_id")]  # columns whose name ends with "_id"
    id_cols.update([(col, 1.0) if col in id_cols else (col, 0.95) for col in col_ends_with_id])

    id_cols_above_threshold = {key: value for key, value in id_cols.items() if value >= threshold}
    return id_cols_above_threshold
