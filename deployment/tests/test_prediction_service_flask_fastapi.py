import json
import logging
import multiprocessing
import time

import pytest
import requests
import uvicorn

from ml_project.historic_data_retrieval import retrieve_historic_data
from ml_project.production_data_retrieval import retrieve_production_data

logger = logging.getLogger('standard')

url_by_framework = {
    'fastapi': "http://127.0.0.1:8000/predict",
    'flask': "http://127.0.0.1:5000/predict",
}


@pytest.fixture(params=['fastapi', 'flask'])
def server_setup_config(request, config_base):

    api_framework = request.param

    config_api = config_base.set_value("prediction_service_url", url_by_framework[api_framework])

    if api_framework == "fastapi":
        from deployment.prediction_service_fastapi import app as fastapi_app

        #fastapi_app = get_fastapi_app(config_api)
        process = multiprocessing.Process(target=uvicorn.run, args=(fastapi_app,))
        process.daemon = True # daemonize process to assure that child process stops if parent testing process stops unexpectedly
        process.start()

    elif api_framework == "flask":
        from deployment.prediction_service_flask import app as flask_app

        #flask_app = get_flaskapi_app(config_api)
        process = multiprocessing.Process(target=flask_app.run, kwargs={'host': '0.0.0.0'})
        process.daemon = True # daemonize process to assure that child process stops if parent testing process stops unexpectedly
        process.start()

    else:
        raise(Exception(f"Not supported api_framework parameter provided: {api_framework}"))

    time.sleep(3) # wait a bit to let the server start

    yield config_api

    # teardown
    process.terminate()



def test_predict(server_setup_config):

    historic_data = retrieve_historic_data(server_setup_config)

    for row_index, row in historic_data.iterrows():

        if row_index in [0]:
            logger.info(row_index, len(historic_data))

            json_string = json.dumps(row.to_dict())

            # Production data retrieval
            production_data = retrieve_production_data(server_setup_config,
                                                       json_string=json_string,
                                                       )

            data_dict = production_data.iloc[0].to_dict()

            response = requests.post(server_setup_config.prediction_service_url, json=data_dict)

            logger.info("Response:")
            logger.info(response.content)

            assert response.status_code == 200