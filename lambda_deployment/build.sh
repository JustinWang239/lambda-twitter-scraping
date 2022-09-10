#!/bin/bash
docker build -f Dockerfile -t trained_model .
docker tag trained_model 131100711312.dkr.ecr.ca-central-1.amazonaws.com/twitter-classification-roberta:latest
docker push 131100711312.dkr.ecr.ca-central-1.amazonaws.com/twitter-classification-roberta:latest