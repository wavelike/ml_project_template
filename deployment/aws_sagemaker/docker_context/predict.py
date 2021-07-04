import json
import os

import pandas as pd

from ml_project.data_validation import ProductionData, validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.modelling_process.data_processing import get_processed_data
from ml_project.prediction_process import get_predictions
from ml_project.production_data_retrieval import process_production_input_data_into_raw_data


def _get_predictions(config, production_data, model_objects):
    print("Files in dir (preds): ", os.listdir())
    print("Workdir (preds): ", os.getcwd())

    ######
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
    model, preprocessing_objects, _ = model_objects
    data_x_prepared, _, _ = get_processed_data(config, preprocessing_objects, data_x)
    predictions, prediction_probas = get_predictions(config, model, data_x_prepared)
    ###
    ######

    return predictions, prediction_probas


def predict(data_array, model_objects):

    # validate and parse request body data
    print("inputting stuff: ", data_array, type(data_array), data_array.shape)
    print("Value: ", data_array)
    print("formatted: ", json.loads(str(data_array).replace("'", '''"''')))

    data_dict = json.loads(str(data_array).replace("'", '''"'''))

    config = model_objects[2]

    data_dict = vars(ProductionData(**data_dict))

    data = pd.DataFrame.from_dict([data_dict])

    prediction = _get_predictions(config, production_data=data, model_objects=model_objects)

    prediction = {
        'label': int(prediction[0].values[0]),
        'proba': prediction[1].values.tolist()
    }

    return json.dumps(prediction)
