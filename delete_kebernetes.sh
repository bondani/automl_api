#!/bin/sh

kubectl delete -f k_api_service.yaml
kubectl delete -f k_api_deployment.yaml
kubectl delete -f k_rq_deployment.yaml
kubectl delete -f k_mongo_service.yaml
kubectl delete -f k_redis_service.yaml
kubectl delete -f k_redis_deployment.yaml
kubectl delete -f k_uploads_pvc.yaml
kubectl delete -f k_uploads_pv.yaml

