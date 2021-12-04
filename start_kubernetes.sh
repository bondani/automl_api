#!/bin/sh

kubectl apply -f k_uploads_pv.yaml
kubectl apply -f k_uploads_pvc.yaml
kubectl apply -f k_redis_deployment.yaml
kubectl apply -f k_redis_service.yaml
kubectl apply -f k_mongo_service.yaml
kubectl apply -f k_rq_deployment.yaml
kubectl apply -f k_api_deployment.yaml
kubectl apply -f k_api_service.yaml