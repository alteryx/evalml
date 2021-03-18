import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import BaseCrossValidator


class TrainingValidationSplit(BaseCrossValidator):
    """Split the training data into training and validation sets"""

    def __init__(self, test_size=None, train_size=None, shuffle=False, stratify=None, random_seed=0):
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
        self.test_size = test_size
        self.train_size = train_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_seed = random_seed

    @staticmethod
    def get_n_splits():
        """Returns the number of splits of this object"""
        return 1

    def split(self, X, y=None):
        """Divides the data into training and testing sets

            Arguments:
                X (pd.DataFrame): Dataframe of points to split
                y (pd.Series): Series of points to split

            Returns:
                list: Indices to split data into training and test set
        """
        train, test = train_test_split(np.arange(X.shape[0]), test_size=self.test_size, train_size=self.train_size, shuffle=self.shuffle, stratify=self.stratify, random_state=self.random_seed)
        return iter([(train, test)])
