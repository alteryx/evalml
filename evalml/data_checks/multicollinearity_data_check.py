
from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)
from woodwork import mutual_information

class MulticollinearityDataCheck(DataCheck):
    """Check if any set features are likely to be multicollinear."""

    def __init__(self, threshold=1.0):
        """Check if any set of features are likely to be multicollinear.

        Arguments:
            threshold (float): The threshold to be considered. Defaults to 1.0.
        """
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1, inclusive.")
        self.threshold = threshold

    def validate(self, X, y=None):
        """Check if any set of features are likely to be multicollinear.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): The input features to check

        Returns:
            dict: A dictionary of features with column name or index and their probability of being ID columns
        """
        messages = {
            "warnings": [],
            "errors": []
        }
        
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        mutual_info = X.mutual_information()
        return messages
