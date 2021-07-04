from typing import Any, Tuple

import pandas as pd
import requests

from ml_project.config import Config
from ml_project.data_validation import ProductionData


def get_predictions(config: Config, model: Any, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:

    predictions = pd.Series(model.predict(data), index=data.index)
    prediction_probas = pd.DataFrame(model.predict_proba(data), index=data.index)

    return predictions, prediction_probas


def get_server_predictions(config: Config, data: pd.Series) -> Tuple[pd.Series, pd.Series]:

    # validate and parse data
    data_dict = vars(ProductionData(**data.to_dict()))

    if config.prediction_service_url is not None:
        response = requests.post(config.prediction_service_url, json=data_dict)
    else:
        raise(Exception("No 'config.prediction_service_url' provided"))

    response_dict = response.json()

    predictions = response_dict['label']
    prediction_probas = response_dict['probas']

    return predictions, prediction_probas