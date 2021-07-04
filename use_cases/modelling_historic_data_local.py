import logging

import mlflow
from mlflow import log_metrics

from ml_project.data_validation import validate_engineered_data_as_batch, validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.evaluation import evaluate_model, evaluate_predictions
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.historic_data_retrieval import process_historic_data_into_raw_data, retrieve_historic_data
from ml_project.ml_monitoring import create_drift_dashboard
from ml_project.model_export import export_model_artifacts
from ml_project.modelling_process.modelling_process import get_optimised_model, get_processed_data, split_features_and_target, split_into_modelling_and_holdout_data
from ml_project.prediction_process import get_predictions
from ml_project.utils import set_mlflow_experiment, setup_logging
from use_cases.use_case_config import config

logger = logging.getLogger('standard')


def execute_modelling_historic_data_local(config):

    ######
    ### setup the run
    logger.info("Setup mlflow experiment")
    set_mlflow_experiment(experiment_name=config.mlflow_experiment)
    ###
    ######

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
    engineered_data, engineered_feature_columns = execute_feature_engineering(config, raw_data, feature_processes)
    validate_engineered_data_per_instance(engineered_data[engineered_feature_columns])
    ###
    ######

    ######
    ### Modelling process
    logger.info("Start modelling process")
    validate_engineered_data_as_batch(config, engineered_data)
    modelling_data, holdout_test_data = split_into_modelling_and_holdout_data(config, engineered_data)
    model, preprocessing_objects, modelling_data_x_prepared, modelling_data_y = get_optimised_model(config, modelling_data)
    modelling_data_predictions, modelling_data_prediction_probas = get_predictions(config, model, modelling_data_x_prepared)
    ###
    ######

    ######
    ### Evaluate final model on holdout test data
    logger.info("Evaluate holdout_test predictions and final model")

    holdout_test_data_x, holdout_test_data_y = split_features_and_target(config, holdout_test_data)
    holdout_test_data_x_prepared, _, _ = get_processed_data(config, preprocessing_objects, holdout_test_data_x)
    holdout_test_predictions, holdout_test_prediction_probas = get_predictions(config, model, holdout_test_data_x_prepared)
    metrics = evaluate_predictions(config, holdout_test_data_y, holdout_test_predictions, holdout_test_prediction_probas)
    log_metrics(metrics)

    evaluate_model(config, model, preprocessing_objects.features, holdout_test_data_x_prepared)
    ###
    ######

    ######
    ### Model export and logging
    logger.info("Export and log model objects")

    if config.export_model_artifacts:
        model_export = export_model_artifacts(config, model, preprocessing_objects)

        mlflow.sklearn.log_model(model_export, 'model')
        logger.info(f"Model saved in run {mlflow.active_run().info.run_uuid}")
    ###
    ######

    ######
    ### ML monitoring for predictions and drift detection
    logger.info("Create ml monitoring report")
    create_drift_dashboard(modelling_data, modelling_data_predictions, modelling_data_prediction_probas,
                           holdout_test_data, holdout_test_predictions, holdout_test_prediction_probas,
                           config.target_col, config.cont_cols, config.drift_dashboard_filepath)
    ###
    ######

if __name__ == '__main__':

    setup_logging('standard')

    execute_modelling_historic_data_local(config)