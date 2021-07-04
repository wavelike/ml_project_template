import json
import logging

import pytest
import requests

from ml_project.data_validation import ProductionData
from ml_project.historic_data_retrieval import retrieve_historic_data
from ml_project.production_data_retrieval import retrieve_production_data

logger = logging.getLogger('standard')

# To succeed with this local test, start the docker container created via 'bash executables cloud_setup.sh' manually in order to start the local prediction server
# sudo docker run -p 8080:8080 mltemplate_lambda
# prediction_service_url="http://localhost:8080/2015-03-31/functions/function/invocations"

url_lambda_local = "http://localhost:8080/2015-03-31/functions/function/invocations"


@pytest.fixture()
def config_lambda_local(config_base):

    return config_base.set_value("prediction_service_url", url_lambda_local)


def test_predict(config_lambda_local):

    historic_data = retrieve_historic_data(config_lambda_local)

    for row_index, row in historic_data.iterrows():

        if row_index in [0]:

            json_string = json.dumps(row.to_dict())

            # Production data retrieval
            production_data = retrieve_production_data(config_lambda_local, json_string=json_string,)

            data_dict = production_data.iloc[0].to_dict()
            data_dict = vars(ProductionData(**data_dict))

            try:
                response = requests.post(config_lambda_local.prediction_service_url, data=json.dumps(data_dict))

                logger.info("Response:")
                logger.info(response.content)

            except requests.exceptions.RequestException as error:
                logger.info(f"Lambda invocation error: {error}")
                response = None

            assert response.status_code == 200

