import numpy as np
import pandas as pd
import pytest

from ml_project.config import Config
from ml_project.prediction_process import get_predictions


@pytest.fixture
def data():

    return pd.DataFrame({
        'cont1': [0, 1, 2, 3],
        'cat1': ['A', 'B', 'B', 'C'],
        'cat2': ['text1', 'text2', 'text2', 'text2'],
        'target': [0, 0, 1, 1]
    })

@pytest.fixture
def config() -> Config:

    return Config(
        historic_or_production_data='historic',
        local_or_deployed='local',
        target_col='target',
        cont_cols=['cont1'],
        cat_cols=['cat1', 'cat2'],
        aux_cols=[],
        data_filepath="", # not relevant for this test
    )

class MockedModel():

    def predict(self, data):
        # mockup prediction behaviour by returning random labels of 0 or 1

        predictions = np.random.choice([0, 1], size=len(data))

        return predictions

    def predict_proba(self, data):
        # mockup prediction behaviour by returning random values between 0 and 1

        prediction_probas = np.random.rand(data.shape[0], 2)

        return prediction_probas

@pytest.fixture
def model():

    return MockedModel()


def test_categorical_encoding(model, data):

    predictions, prediction_probas = get_predictions(config, model, data)

    predictions_have_correct_shape = predictions.shape == (data.shape[0],)
    prediction_probas_have_correct_shape = prediction_probas.shape == (data.shape[0], 2)

    assert all([predictions_have_correct_shape, prediction_probas_have_correct_shape])
