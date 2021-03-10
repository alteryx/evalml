
from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils import infer_feature_types


class MulticollinearityDataCheck(DataCheck):
    """Check if any set features are likely to be multicollinear."""

    def __init__(self, threshold=0.9):
        """Check if any set of features are likely to be multicollinear.

        Arguments:
            threshold (float): The threshold to be considered. Defaults to 0.9.
        """
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1, inclusive.")
        self.threshold = threshold

    def validate(self, X, y=None):
        """Check if any set of features are likely to be multicollinear.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): The input features to check

        Returns:
            dict: dict with a DataCheckWarning if there are any potentially multicollinear columns.

        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        mutual_info_df = X.mutual_information()
        if mutual_info_df.empty:
            return results
        above_threshold = mutual_info_df.loc[mutual_info_df['mutual_info'] >= self.threshold]
        correlated_cols = [(col_1, col_2) for col_1, col_2 in zip(above_threshold['column_1'], above_threshold['column_2'])]
        if correlated_cols:
            warning_msg = "Columns are likely to be correlated: {}"
            results["warnings"].append(DataCheckWarning(message=warning_msg.format(correlated_cols),
                                                        data_check_name=self.name,
                                                        message_code=DataCheckMessageCode.IS_MULTICOLLINEAR,
                                                        details={"columns": correlated_cols}).to_dict())
        return results
