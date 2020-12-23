#!/bin/bash
# Make sure you have MFA authenticated before running this script

AWS_ACCOUNT_ID=321459187557

docker build . -t evalml_dask &&

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com &&
docker tag evalml_dask:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/evalml_dask:latest &&
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/evalml_dask:latest
