import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor


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

    corrs = {label: abs(y.corr(col)) for label, col in X.iteritems() if abs(y.corr(col)) >= threshold}
    return corrs


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


def detect_correlation(X, threshold=.90):
    """Check if correlation exists between features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the correlation threshold to be considered correlated. Defaults to .95.

    Currently only supports checking between numeric-numeric and categorical-categorical features

    Returns:
        A dictionary mapping potentially correlated features and their corresponding correlation coefficient
    """
    correlated = {}
    correlated.update(detect_categorical_correlation(X))
    correlated.update(detect_collinearity(X))
    return correlated


def detect_categorical_correlation(X, threshold=.95):
    """Check if correlation exists between categorical features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the correlation threshold to be considered correlated. Defaults to .95.

    Returns:
        A dictionary mapping potentially collinear features and their corresponding correlation coefficient

    Example:
        >>> X = {'corr_1': [1, 1, 2, 3, 1, 2, 3, 4],
        ...      'corr_2': ['a', 'a', 'b', 'c', 'a', 'b', 'c', 'd'],
        ...      'corr_3': ['w', 'w', 'x', 'y', 'w', 'x', 'y', 'z'],
        ...      'not_corr': [1, 1, 4, 3, 1, 3, 3, 1]}
        >>> X = pd.DataFrame(data=X)
        >>> for col in X:
        ...     X[col] = X[col].astype('category')
        >>> detect_categorical_correlation(X, 0.9).keys() # doctest: +NORMALIZE_WHITESPACE
        dict_keys([('corr_1', 'corr_2'), ('corr_1', 'corr_3'), ('corr_2', 'corr_3')])
    """
    def cramers_v_bias_corrected(confusion_matrix):
        """ Calculate Cramer's V statistic for categorial-categorial correlation with bias correction."""
        chi2 = scipy_stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()  # grand total of observations
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - np.square(r - 1) / (n - 1)
        kcorr = k - np.square(k - 1) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    # only select categorical features
    X = X.select_dtypes(include=['category'])

    cramers_corr = {}
    num_cols = X.shape[1]
    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            # only calculate Cramer's V for upper triangle since Cramer's V produces symmetric scores
            confusion_matrix = pd.crosstab(X.iloc[:, i], X.iloc[:, j])
            col_names = (X.columns[i], X.columns[j])
            cramers_v = cramers_v_bias_corrected(confusion_matrix)
            cramers_corr.update({col_names: cramers_v})
    out = {key: value for (key, value) in cramers_corr.items() if value >= threshold}
    return out


def detect_collinearity(X, threshold=.95):
    """Check if collinearity exists.

    Currently only supports numeric features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the correlation threshold to be considered correlated. Defaults to .95.

    Returns:
        A dictionary mapping potentially collinear features and their corresponding correlation coefficient

    Example:
        >>> col = pd.Series([1, 0, 2, 3, 4])
        >>> X = pd.DataFrame({'col_1': col,
        ...                   'col_2': col*3,
        ...                   'col_3': ~col,
        ...                   'col_4': col/2,
        ...                   'col_5': col+1,
        ...                   'not_collinear': [0, 1, 0, 0, 0]})
        >>> detect_collinearity(X) # doctest: +NORMALIZE_WHITESPACE
        {('col_1', 'col_2'): 1.0, ('col_1', 'col_3'): 1.0, ('col_1', 'col_4'): 1.0,
         ('col_1', 'col_5'): 1.0, ('col_2', 'col_3'): 1.0, ('col_2', 'col_4'): 1.0,
         ('col_2', 'col_5'): 1.0, ('col_3', 'col_4'): 1.0, ('col_3', 'col_5'): 1.0, ('col_4', 'col_5'): 1.0}
    """
    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    corrs = X.corr().abs()
    corrs = corrs.mask(np.tril(np.ones(corrs.shape)).astype(bool)).stack()
    out = {key: value for (key, value) in corrs.items() if value >= threshold}
    return out


def detect_multicollinearity(X, threshold=5):
    """Check if multicollinearity exists amongst numerical features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the VIF threshold to use to determine multicollinearity. Defaults to 5

    Returns:
        A dictionary of features with VIF scores greater than threshold mapped to their corresponding VIF score

    Example:
        >>> col = pd.Series([1, 0, 2, 3, 4])
        >>> X = pd.DataFrame({'col_1': col,
        ...                   'col_2': col*3,
        ...                   'col_3': ~col,
        ...                   'col_4': col/2,
        ...                   'col_5': col+1,
        ...                   'not_mc_col': [0, 1, 0, 0, 0]})
        >>> detect_multicollinearity(X)
        {'col_1': inf, 'col_2': inf, 'col_3': inf, 'col_4': inf, 'col_5': inf}
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)
    if len(X.columns) == 0:
        return {}

    multicollinear_cols = {}
    X = X.dropna(axis=1, how='any')
    X = X.assign(const=1)  # since variance_inflation_factor doesn't add intercept
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    vif = vif[vif >= threshold]
    multicollinear_cols = vif.to_dict()
    return multicollinear_cols


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
    """Check if any of the features are ID columns. Currently performs these simple checks:

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
