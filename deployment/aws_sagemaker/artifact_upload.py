import boto3
from botocore.exceptions import ClientError


def upload_to_s3(source_path, key_name):

    model_filepath = "deployment/aws_sagemaker/docker_context/model_titanic.tar.gz"
    bucket_name = "mltemplate"
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(model_filepath, bucket_name, 'model_titanic.tar.gz')  # , object_name)
    except ClientError as e:
        print(e)

    print("Uploaded model objects to S3 bucket")
