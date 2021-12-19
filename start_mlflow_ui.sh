#!/bin/bash

MLFLOW_PORT=$(grep 'mlflow_port' configs/dev.yml | awk '{print $2}')

mlflow ui --port $MLFLOW_PORT