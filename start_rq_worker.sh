#!/bin/bash

REDIS_HOST=$(grep 'redis_host' configs/dev.yml | awk '{print $2}')
REDIS_PORT=$(grep 'redis_port' configs/dev.yml | awk '{print $2}')

rqworker --with-scheduler -u redis://$REDIS_HOST:$REDIS_PORT