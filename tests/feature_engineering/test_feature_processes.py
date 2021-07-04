

import pandas as pd
import pytest

from ml_project.feature_engineering.feature_processes import Feature1Feature2Ratio, Feature1Feature2Sum


@pytest.fixture
def data():

    return pd.DataFrame({
        'a': [0, 1, 2, 3],
        'b': [10, 11, 12, 13]
    })


def test_feature1_feature2_sum(data):

    expected_result = [10, 12, 14, 16]

    feature_process = Feature1Feature2Sum(feature1_col='a',
                                          feature2_col='b',
                                          column_name='a_b_sum'
                                          )

    data = feature_process.execute(data)

    assert (data['a_b_sum'] == pd.Series(expected_result)).all()


def test_feature1_feature2_ratio(data):

    expected_result = [0, 1/11, 2/12, 3/13]

    feature_process = Feature1Feature2Ratio(feature1_col='a',
                                          feature2_col='b',
                                          column_name='a_b_ratio'
                                          )

    data = feature_process.execute(data)

    assert (data['a_b_ratio'] == pd.Series(expected_result)).all()
