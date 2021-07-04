#!/bin/bash

region=eu-west-1
image_tag=mltemplate_lambda

# if input parameter exist
if [ $# -eq 1 ]
then
  AWS_ACCOUNT_ID=$1
fi

# login to ecr and send password to docker in order to connect services
aws ecr get-login-password --region $region | sudo docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$region.amazonaws.com

# tag image
sudo docker tag $image_tag "${AWS_ACCOUNT_ID}".dkr.ecr.$region.amazonaws.com/$image_tag

# push image to ecr
sudo docker push "${AWS_ACCOUNT_ID}".dkr.ecr.$region.amazonaws.com/$image_tag