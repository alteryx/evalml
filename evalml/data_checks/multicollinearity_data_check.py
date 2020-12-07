
from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


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
        mutual_info_df = X.mutual_information()
        # zero or one case?
        if mutual_info_df.empty:
            return messages
        above_threshold = mutual_info_df.loc[mutual_info_df['mutual_info'] >= self.threshold]
        correlated_cols = [(col_1, col_2) for col_1, col_2 in zip(above_threshold['column_1'], above_threshold['column_2'])]
        if correlated_cols:
            warning_msg = "Columns are likely to be correlated: {}"
            messages["warnings"].append(DataCheckWarning(message=warning_msg.format(correlated_cols),
                                                         data_check_name=self.name,
                                                         message_code=DataCheckMessageCode.IS_MULTICOLLINEAR,
                                                         details={"columns": correlated_cols}).to_dict())
        return messages
