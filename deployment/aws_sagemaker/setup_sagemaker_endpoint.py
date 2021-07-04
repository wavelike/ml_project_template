import logging
import os
import subprocess

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.predictor import Predictor
from sagemaker.sklearn.model import SKLearnModel

from ml_project.utils import setup_logging

logger = logging.getLogger('standard')


def upload_to_s3(source_path, bucket_name, key_name):

    s3_client = boto3.client('s3')

    try:
        response = s3_client.upload_file(source_path, bucket_name, key_name)
    except ClientError as e:
        print(e)

    logger.info("Uploaded model objects to S3 bucket")


def prepare_endpoint(instance_type, model_artifact_path_pkl, model_artifact_path_zip):

    logger.info("Instance type = " + instance_type)

    if instance_type == "local":

        sagemaker_session = sagemaker.LocalSession()

        try:
            if subprocess.call("nvidia-smi") == 0:
                ## Set type to GPU if one is present
                instance_type = "local_gpu"
        except:
            pass

        artifact_path_deployed = f"file://{model_artifact_path_pkl}"

    else:
        sagemaker_session = sagemaker.Session()

        artifact_path_deployed = "s3://mltemplate/model_titanic.tar.gz"

        upload_to_s3(source_path=model_artifact_path_zip, bucket_name="mltemplate", key_name='model_titanic.tar.gz')

    # Either get role ARN from current running AWS instance (e.g. Sagemaker studio notebook) or specify the ExecutionRole manually
    # If you do not have a Sagemaker Execution Role specified yet, follow the steps on https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
    try:
        role_arn = sagemaker.get_execution_role()
    except:
        iam = boto3.client('iam')
        role_arn = iam.get_role(RoleName=os.environ['SAGEMAKER_EXECUTION_ROLE'])['Role']['Arn'] # TODO: Take this from environment variable as it is private data

    logger.info(f"Role ARN: {role_arn}")

    return sagemaker_session, artifact_path_deployed, role_arn, instance_type


def start_endpoint(instance_type, model_artifact_path_pkl, model_artifact_path_zip):

    sagemaker_session, artifact_path_deployed, role_arn, instance_type = prepare_endpoint(instance_type, model_artifact_path_pkl, model_artifact_path_zip)

    # Wrap existing model artifacts in a Sagemaker Estimator object
    titanic_estimator = SKLearnModel(model_data=artifact_path_deployed,
                                     source_dir="deployment/aws_sagemaker/docker_context",
                                     entry_point="serve.py",
                                     py_version="py3",
                                     role=role_arn,
                                     framework_version='0.23-1',
                                     )

    # deploy the estimator
    titanic_predictor: Predictor = titanic_estimator.deploy(initial_instance_count=1, instance_type=instance_type)

    endpoint_name = titanic_predictor.endpoint_name
    logger.info(f"Endpoint name: {endpoint_name}")


    # titanic_predictor.delete_endpoint()


if __name__ == '__main__':

    setup_logging('standard')

    #instance_type = "local"
    instance_type = "ml.t2.medium"

    model_artifact_path_pkl = 'deployment/aws_sagemaker/docker_context/model_titanic.pkl'
    model_artifact_path_zip = 'deployment/aws_sagemaker/docker_context/model_titanic.tar.gz'

    start_endpoint(instance_type, model_artifact_path_pkl, model_artifact_path_zip)


