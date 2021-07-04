import json
import logging

import sagemaker
from sagemaker import Predictor
from sagemaker.deserializers import NumpyDeserializer
from sagemaker.serializers import NumpySerializer

from ml_project.data_validation import ProductionData
from ml_project.historic_data_retrieval import retrieve_historic_data
from ml_project.production_data_retrieval import retrieve_production_data
from ml_project.utils import setup_logging

logger = setup_logging('standard')

#endpoint_name = 'sagemaker-scikit-learn-2021-06-16-19-36-48-272'
#endpoint_name = 'sagemaker-scikit-learn-2021-06-16-22-43-47-125'
#endpoint_name = 'sagemaker-scikit-learn-2021-06-17-11-10-01-953'
#endpoint_name = 'sagemaker-scikit-learn-2021-06-26-15-33-08-400'
endpoint_name = 'sagemaker-scikit-learn-2021-06-26-16-45-44-681'
endpoint_name = 'sagemaker-scikit-learn-2021-06-27-13-28-09-323'
endpoint_name = 'sagemaker-scikit-learn-2021-06-27-13-30-24-773'
endpoint_name = 'sagemaker-scikit-learn-2021-06-27-14-11-52-366'
endpoint_name = 'sagemaker-scikit-learn-2021-07-03-16-09-07-525'

sagemaker_session = sagemaker.LocalSession()
#sagemaker_session = sagemaker.Session()

def test_predict(config_base):

    historic_data = retrieve_historic_data(config_base)

    for row_index, row in historic_data.iterrows():

        if row_index in [0]:

            json_string = json.dumps(row.to_dict())

            # Production data retrieval
            production_data = retrieve_production_data(config_base,
                                                       json_string=json_string,
                                                       )

            data_dict = production_data.iloc[0].to_dict()
            data_dict = vars(ProductionData(**data_dict))

            titanic_predictor = Predictor(endpoint_name,
                                           sagemaker_session=sagemaker_session,
                                           serializer=NumpySerializer(),
                                           deserializer=NumpyDeserializer(),
                                           )

            response_array = titanic_predictor.predict(data_dict)

            response_dict = json.loads(str(response_array).replace("'", '''"'''))

            logger.info("Response:")
            logger.info(response_dict)

