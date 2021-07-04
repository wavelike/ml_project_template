import json
import logging

import pytest
import requests

from ml_project.data_validation import ProductionData
from ml_project.historic_data_retrieval import retrieve_historic_data
from ml_project.production_data_retrieval import retrieve_production_data
from ml_project.utils import setup_logging

logger = setup_logging('standard')

app_engine_server_url = "https://deployment-316521.ey.r.appspot.com/predict"


@pytest.fixture()
def config_app_engine(config_base):

    return config_base.set_value("prediction_service_url", app_engine_server_url)


def test_predict(config_app_engine):

    historic_data = retrieve_historic_data(config_app_engine)

    for row_index, row in historic_data.iterrows():

        if row_index in [0]:

            json_string = json.dumps(row.to_dict())

            # Production data retrieval
            production_data = retrieve_production_data(config_app_engine,
                                                       json_string=json_string,
                                                       )


            data_dict = production_data.iloc[0].to_dict()
            data_dict = vars(ProductionData(**data_dict))

            response = requests.post(config_app_engine.prediction_service_url, json=data_dict)

            logger.info("Response:")
            logger.info(response.content)

            assert response.status_code == 200