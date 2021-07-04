import logging

from ml_project.config import Config
from ml_project.data_validation import validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.evaluation import evaluate_predictions
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.historic_data_retrieval import process_historic_data_into_raw_data, retrieve_historic_data
from ml_project.model_export import load_model_artifacts
from ml_project.modelling_process.data_processing import get_processed_data
from ml_project.modelling_process.modelling_process import split_features_and_target
from ml_project.prediction_process import get_predictions
from ml_project.utils import setup_logging
from use_cases.use_case_config import config

logger = logging.getLogger('standard')


def execute_predicting_historical_data_local(config: Config):

    ######
    ### Historic data retrieval
    logger.info("Retrieve historical data")
    historic_data = retrieve_historic_data(config)
    raw_data = process_historic_data_into_raw_data(config, historic_data)
    validate_raw_data_per_instance(raw_data)
    ###
    ######

    ######
    ### Feature engineering
    logger.info("Run feature engineering")
    feature_processes = get_feature_processes(config)
    test_data, engineered_feature_columns = execute_feature_engineering(config, raw_data, feature_processes)
    validate_engineered_data_per_instance(test_data[engineered_feature_columns])
    ###
    ######

    ######
    ### Loading of model artifacts
    logger.info("Loading model artifacts")
    model, preprocessing_objects, _ = load_model_artifacts(model_objects_filepath=config.export_filepath)
    ###
    ######

    ######
    ### Evaluate final model on holdout test data
    logger.info("Evaluate holdout_test predictions and final model")
    test_data_x, test_data_y = split_features_and_target(config, test_data)
    test_data_x_prepared, _, _ = get_processed_data(config, preprocessing_objects, test_data_x)
    test_predictions, test_prediction_probas = get_predictions(config, model, test_data_x_prepared)
    evaluate_predictions(config, test_data_y, test_predictions, test_prediction_probas)
    ###
    ######


if __name__ == '__main__':

    setup_logging('standard')

    execute_predicting_historical_data_local(config)