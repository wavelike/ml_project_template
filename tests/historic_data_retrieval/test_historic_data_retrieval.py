import pandas as pd
import pytest

from ml_project.historic_data_retrieval import retrieve_from_parquet


@pytest.fixture
def data():

    return pd.DataFrame({
        'cont1': [0, 1, 2, 3],
        'cat1': ['A', 'B', 'B', 'C'],
        'cat2': ['text1', 'text2', 'text2', 'text2']
    })


@pytest.fixture
def filepath_parquet(tmpdir, data):

    filepath = tmpdir.join('data.parquet')
    data.to_parquet(filepath)

    return str(filepath)


def test_retrieve_from_parquet(filepath_parquet, data):

    retrieved_data = retrieve_from_parquet(filepath_parquet)

    assert retrieved_data.equals(data)