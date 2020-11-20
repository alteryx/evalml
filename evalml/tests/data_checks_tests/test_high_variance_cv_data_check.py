import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    HighVarianceCVDataCheck
)

high_variance_data_check_name = HighVarianceCVDataCheck.name


def test_high_variance_cv_data_check_invalid_threshold():
    with pytest.raises(ValueError, match="needs to be greater than 0."):
        HighVarianceCVDataCheck(threshold=-0.1).validate(pipeline_name='LogisticRegressionPipeline', cv_scores=pd.Series([0, 1, 1]))


def test_high_variance_cv_data_check():
    high_variance_cv = HighVarianceCVDataCheck()

    assert high_variance_cv.validate(pipeline_name='LogisticRegressionPipeline', cv_scores=[0, 0, 0]) == {"warnings": [], "errors": []}
    assert high_variance_cv.validate(pipeline_name='LogisticRegressionPipeline', cv_scores=[1, 1, 1]) == {"warnings": [], "errors": []}
    assert high_variance_cv.validate(pipeline_name='LogisticRegressionPipeline', cv_scores=pd.Series([1, 1, 1])) == {"warnings": [], "errors": []}
    assert high_variance_cv.validate(pipeline_name='LogisticRegressionPipeline', cv_scores=pd.Series([0, 1, 2, 3])) == {
        "warnings": [DataCheckWarning(message="High coefficient of variation (cv >= 0.2) within cross validation scores. LogisticRegressionPipeline may not perform as estimated on unseen data.",
                                      data_check_name=high_variance_data_check_name,
                                      message_code=DataCheckMessageCode.HIGH_VARIANCE).to_dict()],
        "errors": []
    }


def test_high_variance_cv_data_check_empty_nan():
    high_variance_cv = HighVarianceCVDataCheck()
    assert high_variance_cv.validate(pipeline_name='LogisticRegressionPipeline', cv_scores=pd.Series([0, 1, np.nan, np.nan])) == {
        "warnings": [DataCheckWarning(message="High coefficient of variation (cv >= 0.2) within cross validation scores. LogisticRegressionPipeline may not perform as estimated on unseen data.",
                                      data_check_name=high_variance_data_check_name,
                                      message_code=DataCheckMessageCode.HIGH_VARIANCE).to_dict()],
        "errors": []
    }


def test_high_variance_cv_data_check_negative():
    high_variance_cv = HighVarianceCVDataCheck()
    assert high_variance_cv.validate(pipeline_name='LogisticRegressionPipeline', cv_scores=pd.Series([0, -1, -1, -1])) == {
        "warnings": [DataCheckWarning(message="High coefficient of variation (cv >= 0.2) within cross validation scores. LogisticRegressionPipeline may not perform as estimated on unseen data.",
                                      data_check_name=high_variance_data_check_name,
                                      message_code=DataCheckMessageCode.HIGH_VARIANCE).to_dict()],
        "errors": []
    }
