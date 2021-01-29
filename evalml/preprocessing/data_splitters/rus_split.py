from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import BaseCrossValidator
from imblearn.under_sampling import RandomUnderSampler

class RandomUnderSamplerSplit(BaseCrossValidator):
    """Split the training data into training and validation sets. Uses RandomUnderSampler to balance the training data,
       but keeps the validation data the same"""

    def __init__(self, sampling_strategy='auto', replacement=False, test_size=None, random_state=0):
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.test_size = test_size
        self.random_state = random_state

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
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data. 
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        rus = RandomUnderSampler(sampling_strategy=self.sampling_strategy, replacement=self.replacement, random_state=self.random_state)
        X_train_resample, y_train_resample = rus.fit_resample(X_train, y_train)
        print("TRAIN", len(X_train), len(X_train_resample))
        print("X_TEST", len(X_test))
        return iter([(X_train_resample, X_test, y_train_resample, y_test)])