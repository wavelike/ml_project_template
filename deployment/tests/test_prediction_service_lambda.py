import json
import logging
import os

import boto3
import botocore

from ml_project.data_validation import ProductionData
from ml_project.historic_data_retrieval import retrieve_historic_data
from ml_project.production_data_retrieval import retrieve_production_data

logger = logging.getLogger('standard')


function_arn = "arn:aws:lambda:eu-west-1:478081143523:function:mltemplate_container"


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


            lambda_client = boto3.client("lambda",
                                         aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
                                         aws_secret_access_key=os.environ.get('SECRET_ACCESS_KEY')
                                         )

            try:
                response = lambda_client.invoke(FunctionName=function_arn,
                                     InvocationType='RequestResponse',
                                     LogType='Tail',
                                     Payload=json.dumps(data_dict)
                                     )
            except botocore.exceptions.ClientError as error:
                logger.error(f"Lambda invocation error: {error}")


            payload = json.loads(response['Payload'].read())

            try:
                body = json.loads(payload['body'])
            except:
                logger.warning("No 'body' key found in 'payload'")
                logger.warning(payload)

            logger.info(body)
