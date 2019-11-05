import pandas as pd
import scipy as sp


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


def detect_numerical_categorical_correlation(num_col, cat_col):
    """ Compares a numerical and a categorical column for correlation using a one-way ANOVA test.
        The null hypothesis is that the mean between samples is the same.
        Thus rejecting the null implies that the num_col correlates with the cat_col as each category has
        a statistically significant difference in mean.

    Args:
        num_col (pd.Series or list) : numerical column
        cat_col (pd.Series or list) : categorical column

    Returns:
        chisquare statistic, p-value
    """
    if not isinstance(num_col, pd.Series):
        num_col = pd.Series(num_col)
    if not isinstance(cat_col, pd.Series):
        cat_col = pd.Series(cat_col)

    num_col = num_col.rename('num')
    cat_col = cat_col.rename('cat')
    df = pd.concat([num_col, cat_col], axis=1)
    num_per_cat = []

    for _, cat_df in df.groupby('cat'):
        num_per_cat.append(cat_df['num'].tolist())

    statistic, pvalue = sp.stats.f_oneway(*num_per_cat)
    return statistic, pvalue


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
