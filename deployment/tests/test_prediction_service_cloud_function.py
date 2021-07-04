import json
import logging

import pytest
import requests

from ml_project.config import Config
from ml_project.data_validation import ProductionData
from ml_project.historic_data_retrieval import retrieve_historic_data
from ml_project.production_data_retrieval import retrieve_production_data
from ml_project.utils import setup_logging

logger = setup_logging('standard')


url_cloud_function = "https://europe-west3-deployment-316521.cloudfunctions.net/api_predict"

@pytest.fixture()
def config_cloud_function(config_base):

    return config_base.set_value("prediction_service_url", url_cloud_function)


def test_predict(config_cloud_function):

    historic_data = retrieve_historic_data(config_cloud_function)

    for row_index, row in historic_data.iterrows():

        if row_index in [0]:

            json_string = json.dumps(row.to_dict())

            # Production data retrieval
            production_data = retrieve_production_data(config_cloud_function,
                                                       json_string=json_string,
                                                       )


            data_dict = production_data.iloc[0].to_dict()
            data_dict = vars(ProductionData(**data_dict))

            response = requests.post(config_cloud_function.prediction_service_url, json=data_dict)

            logger.info("Response:")
            logger.info(response.content)

            assert response.status_code == 200