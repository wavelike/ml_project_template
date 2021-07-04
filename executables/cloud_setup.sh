#!/bin/bash

#cloud_provider='none'
#cloud_provider='heroku'
#cloud_provider='google_app_engine'
#cloud_provider='google_cloud_function'
#cloud_provider='aws_lambda'
cloud_provider='sagemaker'


# cleanup possibly existing files
bash executables/cleanup_project.sh

if [ $cloud_provider = 'heroku' ]
then
  cp deployment/heroku/Procfile .
  cp deployment/prediction_service_fastapi.py . # using fastpi

elif [ $cloud_provider = 'google_app_engine' ]
then
  cp deployment/google_app_engine/app.yaml .
  cp deployment/prediction_service_flask.py ./main.py # using flask
  cp deployment/google_app_engine/.gcloudignore .
  pipenv lock --requirements > requirements.txt

elif [ $cloud_provider = 'google_cloud_function' ]
then
  mkdir -p output/function_deployment
  cp deployment/google_cloud_function/cloud_function.py output/function_deployment/main.py
  pipenv lock --requirements > output/function_deployment/requirements.txt
  cp -r ml_project output/function_deployment/ml_project
  cp project_config.yaml output/function_deployment/project_config.yaml
  mkdir -p output/function_deployment/data/serialised_models
  cp data/serialised_models/* output/function_deployment/data/serialised_models
  cd output/function_deployment || exit
  zip -r function_deployment.zip main.py requirements.txt ml_project data project_config.yaml

elif [ $cloud_provider = 'aws_lambda' ]
then

  cp deployment/aws_lambda/lambda_function.py lambda_function.py
  cp deployment/aws_lambda/Dockerfile .
  cp deployment/aws_lambda/.dockerignore .
  pipenv lock --requirements > requirements.txt
  docker build -t mltemplate_lambda .

  if [ $# -eq 0 ]
  then
    echo The image is built but not pushed to ecr. If you want to, provide the 'AWS_ACCOUNT_ID' as first parameter of the script call
    exit 1

  else
    bash deployment/aws_lambda/docker_image_to_ecr.sh $1
  fi

  # Local run:
  # sudo docker run -p 8080:8080 mltemplate_lambda
  # prediction_service_url="http://localhost:8080/2015-03-31/functions/function/invocations"

elif [ $cloud_provider = 'sagemaker' ]
then
  cp -r ml_project deployment/aws_sagemaker/docker_context
  #pipenv lock --requirements > deployment/aws_sagemaker/docker_context/requirements.txt
  cp data/serialised_models/model_titanic.pkl deployment/aws_sagemaker/docker_context

  cp deployment/aws_sagemaker/setup_sagemaker_endpoint.py setup_sagemaker_endpoint.py

  cd deployment/aws_sagemaker/docker_context
  tar -czf model_titanic.tar.gz model_titanic.pkl
#  tar -czf deployment/aws_sagemaker/docker_context/model_titanic.tar.gz data/serialised_models/model_titanic.pkl

  # to deploy a pre-trained model
  #sudo -E env "PATH=$PATH" pipenv run python setup_sagemaker_endpoint.py
fi

