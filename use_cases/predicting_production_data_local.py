import logging

from ml_project.config import Config
from ml_project.data_validation import validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.model_export import load_model_artifacts
from ml_project.modelling_process.modelling_process import get_processed_data
from ml_project.prediction_process import get_predictions
from ml_project.production_data_retrieval import process_production_input_data_into_raw_data, retrieve_production_data, run_production_simulator
from ml_project.utils import setup_logging
from use_cases.use_case_config import config

logger = logging.getLogger('standard')



def execute_predicting_production_data_local(config: Config, json_string: str):

    ######
    ### Production data retrieval
    logger.info("Retrieve production data")
    production_data = retrieve_production_data(config,
                                               json_string=json_string,
                                               )
    raw_data = process_production_input_data_into_raw_data(config, production_data)
    validate_raw_data_per_instance(raw_data)
    ###
    ######

    ######
    ### Feature engineering
    logger.info("Run feature engineering")
    feature_processes = get_feature_processes(config)
    data_x, engineered_feature_columns = execute_feature_engineering(config, raw_data, feature_processes)
    validate_engineered_data_per_instance(data_x[engineered_feature_columns])
    ###
    ######

    ######
    ### Model loading
    logger.info("Loading model objects")
    model, preprocessing_objects, _ = load_model_artifacts(model_objects_filepath=config.export_filepath)
    ###
    ######

    ######
    ### Retrieve predictions
    logger.info("Evaluate holdout_test predictions")
    data_x_prepared, _, _ = get_processed_data(config, preprocessing_objects, data_x)
    predictions, prediction_probas = get_predictions(config, model, data_x_prepared)
    ###
    ######

    return predictions, prediction_probas


if __name__ == '__main__':

    setup_logging('standard')

    run_production_simulator(config, execute_predicting_production_data_local)
