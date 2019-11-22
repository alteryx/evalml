import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_label_leakage(X, y, threshold=.95):
    """Check if any of the features are highly correlated with the target.

    Currently only supports binary and numeric targets and features

    Args:
        X (pd.DataFrame): The input features to check
        y (pd.Series): the labels
        threshold (float): the correlation threshold to be considered leakage. Defaults to .95

    Returns:
        leakage, dictionary of features with leakage and corresponding threshold

    Example:
        >>> X = pd.DataFrame({
        ...    'leak': [10, 42, 31, 51, 61],
        ...    'x': [42, 54, 12, 64, 12],
        ...    'y': [12, 5, 13, 74, 24],
        ... })
        >>> y = pd.Series([10, 42, 31, 51, 40])
        >>> detect_label_leakage(X, y, threshold=0.8)
        {'leak': 0.8827072320669518}
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
        X (pd.DataFrame) : features
        percent_threshold(float): Require that percentage of null values to be considered "highly-null", defaults to .95

    Returns:
        A dictionary of features with column name or index and their percentage of null values

    Example:
        >>> df = pd.DataFrame({
        ...    'lots_of_null': [None, None, None, None, 5],
        ...    'no_null': [1, 2, 3, 4, 5]
        ... })
        >>> detect_highly_null(df, percent_threshold=0.8)
        {'lots_of_null': 0.8}
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    percent_null = (X.isnull().mean()).to_dict()
    highly_null_cols = {key: value for key, value in percent_null.items() if value >= percent_threshold}
    return highly_null_cols


def detect_outliers(X, random_state=0):
    """ Checks if there are any outliers in a dataframe by using first Isolation Forest to obtain the anomaly score
    of each index and then using IQR to determine score anomalies. Indices with score anomalies are considered outliers.

    Args:
        X (pd.DataFrame): features

    Returns:
        A set of indices that may have outlier data.

    Example:
        >>> df = pd.DataFrame({
        ...     'x': [1, 2, 3, 40, 5],
        ...     'y': [6, 7, 8, 990, 10],
        ...     'z': [-1, -2, -3, -1201, -4]
        ... })
        >>> detect_outliers(df)
        [3]
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    def get_IQR(df, k=2.0):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (k * iqr)
        upper_bound = q3 + (k * iqr)
        return (lower_bound, upper_bound)

    clf = IsolationForest(random_state=random_state, behaviour="new", contamination=0.1)
    clf.fit(X)
    scores = pd.Series(clf.decision_function(X))
    lower_bound, upper_bound = get_IQR(scores, k=2)
    outliers = (scores < lower_bound) | (scores > upper_bound)
    outliers_indices = outliers[outliers].index.values.tolist()
    return outliers_indices


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

    Example:
        >>> df = pd.DataFrame({
        ...     'df_id': [0, 1, 2, 3, 4],
        ...     'x': [10, 42, 31, 51, 61],
        ...     'y': [42, 54, 12, 64, 12]
        ... })
        >>> detect_id_columns(df)
        {'df_id': 1.0}
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
