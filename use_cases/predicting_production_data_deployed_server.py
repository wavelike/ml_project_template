import logging

from ml_project.config import Config
from ml_project.prediction_process import get_server_predictions
from ml_project.production_data_retrieval import retrieve_production_data, run_production_simulator
from ml_project.utils import setup_logging
from use_cases.use_case_config import config

logger = logging.getLogger('standard')


def execute_predicting_production_data_local_server(config: Config, json_string):

    ######
    ### Production data retrieval
    production_data = retrieve_production_data(config,
                                               json_string=json_string,
                                               )
    ###
    ######

    ######
    ### Retrieve predictions
    predictions, prediction_probas = get_server_predictions(config, data=production_data.iloc[0])
    ###
    ######

    return predictions, prediction_probas


if __name__ == '__main__':

    setup_logging('standard')

    run_production_simulator(config, execute_predicting_production_data_local_server)