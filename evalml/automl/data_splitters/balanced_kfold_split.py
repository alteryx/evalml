from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class BalancedKFoldSplit(BaseCrossValidator):
    """Split the training data into training and validation sets"""

    def __init__(self, n_splits=3, shuffle=False, random_state=0):
        """Create a TrainingValidation instance

        Arguments:
            test_size (float): What percentage of data points should be included in the validation
                set. Defalts to the complement of `train_size` if `train_size` is set, and 0.25 otherwise.

            train_size (float): What percentage of data points should be included in the training set.
                Defaults to the complement of `test_size`.

            shuffle (bool): Whether to shuffle the data before splitting. Defaults to True.

            stratify (list): Splits the data in a stratified fashion, using this argument as class labels.
                Defaults to None.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self):
        """Returns the number of splits of this object"""
        return self.n_splits

    def split(self, X, y=None):
        """Divides the data into training and testing sets

            Arguments:
                X (pd.DataFrame): dataframe of points to split
                y (pd.Series): series of points to split

            Returns:
                list: indices to split data into training and test set
        """
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        for train_index, test_index in skf.split(X, y):
            _ = RandomUnderSampler(random_state=self.random_state, replacement=False)
            _ = RandomUnderSampler(random_state=self.random_state, replacement=False)

        #     X_train_resampled, y_train_resampled = rus_train.fit_resample(X[train_index], y[train_index])
        #     X_test_resampled, y_test_resampled = rus_test.fit_resample(X[test_index], y[test_index])

        #     X[train_index] = X_train_resampled
        #     y[train_index] = y_train_resampled

        #     X[test_index] = X_test_resampled
        #     y[train_index] = y_train_resampled

        # return skf.split(X_resampled, y_resampled)
