from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import BaseCrossValidator


class TrainingValidationSplit(BaseCrossValidator):

    def __init__(self, test_size=None, train_size=None, shuffle=True, stratify=None, random_state=0):
        self.test_size = test_size
        self.train_size = train_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_state = random_state

    @staticmethod
    def get_n_splits():
        return 1

    def split(self, X, y=None):
        return train_test_split(X, y, test_size=self.test_size, train_size=self.train_size, shuffle=self.shuffle, stratify=self.stratify, random_state=self.random_state)
