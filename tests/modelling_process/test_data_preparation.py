
import pandas as pd
import pytest

from ml_project.config import Config
from ml_project.modelling_process.data_processing import PreprocessingObjects, apply_categorical_encoding


@pytest.fixture
def data():

    return pd.DataFrame({
        'cont1': [0, 1, 2, 3],
        'cat1': ['A', 'B', 'B', 'C'],
        'cat2': ['text1', 'text2', 'text2', 'text2']
    })

@pytest.fixture
def config():

    return Config(
        historic_or_production_data='historic',
        local_or_deployed='local',
        target_col='',
        cont_cols=['cont1'],
        cat_cols=['cat1', 'cat2'],
        aux_cols=[],
        data_filepath="", # not relevant for this test
    )

@pytest.fixture
def preprocessing_objects():

    return PreprocessingObjects()


def test_categorical_encoding(config, preprocessing_objects, data):

    expected_result = pd.DataFrame({
        'cont1': [0, 1, 2, 3],
        'cat1_A': [1, 0, 0, 0],
        'cat1_B': [0, 1, 1, 0],
        'cat1_C': [0, 0, 0, 1],
        'cat2_text1': [1, 0, 0, 0],
        'cat2_text2': [0, 1, 1, 1],
    })

    data, _, _ = apply_categorical_encoding(config, preprocessing_objects, data)

    assert (data == expected_result).all().all()


