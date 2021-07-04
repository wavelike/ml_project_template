import json
from typing import Tuple

import pandas as pd

from ml_project.config import Config
from ml_project.data_validation import validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.model_export import load_model_artifacts
from ml_project.modelling_process.data_processing import get_processed_data
from ml_project.prediction_process import get_predictions
from ml_project.production_data_retrieval import process_production_input_data_into_raw_data
from ml_project.utils import get_model_artifacts_filepath

# Model loading
model, preprocessing_objects, config = load_model_artifacts(model_objects_filepath=get_model_artifacts_filepath())

def _get_predictions(config: Config, production_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:

    #######
    ### data into raw_data
    raw_data = process_production_input_data_into_raw_data(config, production_data)
    validate_raw_data_per_instance(raw_data)
    ###
    ######

    ######
    ### Feature engineering
    feature_processes = get_feature_processes(config)
    data_x, engineered_feature_columns = execute_feature_engineering(config, raw_data, feature_processes)
    validate_engineered_data_per_instance(data_x[engineered_feature_columns])
    ###
    ######

    ######
    ### Retrieve predictions
    data_x_prepared, _, _ = get_processed_data(config, preprocessing_objects, data_x)
    predictions, prediction_probas = get_predictions(config, model, data_x_prepared)
    ###
    ######

    return predictions, prediction_probas


# TODO: Refactor this when lambda works again
def predict(event, context):

    # validate and parse request body data
    #data_dict = vars(Data(**request.get_json()))
    data_dict = event

    print("data_dict:")
    print(data_dict)

    data = pd.DataFrame.from_dict([data_dict])

    prediction = _get_predictions(config, production_data=data)

    prediction = {
        'label': int(prediction[0].values[0]),
        'proba': prediction[1].values.tolist()
    }

    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event,
        "prediction": prediction
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
