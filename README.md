![docker workflow](https://github.com/bondani/automl_api/actions/workflows/docker-image.yml/badge.svg)

![python workflow](https://github.com/bondani/automl_api/actions/workflows/python-app.yml/badge.svg)


# Запуск приложения с помощью Dockerfile'ов

## 1. Запустить инстанс БД Mongo

```
docker run --name some-mongo -p 57017:27017 mongo
```

## 2. Запустить инстанс Redis

```
docker run --name some-redis -p 56379:6379 --rm -d redis
```

## 3. Создать образ докера веб-приложения и воркера

```
docker build -t automl_api -f docker/api/Dockerfile .
docker build -t automl_rq -f docker/rq/Dockerfile .
```

## 4. Запустить инстанс Prometheus

_Перед запуском необходимо произвести конфигурирование через `configs/prometheus.yml`. Основное: изменить адреса приложений в зависимости от того, в какой подсети они запущены._

```
docker run \
    -p 59090:9090 --name some-prometheus \
    -v ${PWD}/prometheus:/prometheus \
    -v ${PWD}/configs/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

Dashboard Prometheus'а доступен по адресу: `localhost:59090`

## 5. Запустить инстанс AlertManager

```
docker run --name some-alertmanager -d -p 59093:9093 -v ${PWD}/configs/alertmanager.yml:/etc/alertmanager/alertmanager.yml  quay.io/prometheus/alertmanager
```

_В учебных целях в качестве алерта используется величина - среднее количество запросов по времени (минута). Проверить работоспособность можно сделать более 500 запросов на любой /api - эндпоит в течение минуты._

Dashboard AlertManager'а доступен по адресу: `localhost:59093`

## 6. Запусть контейнеры на основе созданных образов

```
docker run -d -p 58080:8080 -v ${PWD}/uploads:/backend/uploads -v ${PWD}/configs:/backend/configs --name some-automl automl_api

docker run -d -v ${PWD}/configs:/backend/configs --name some-rq automl_rq
```

_Собранные образы доступны по названиям: `bondani/automl_api:latest` и `bondani/automl_rq:latest`_

_`https://hub.docker.com/repository/docker/bondani/automl_api` `https://hub.docker.com/repository/docker/bondani/automl_rq`_

# Запуск приложения с помощью docker-compose

```
docker-compose up -d
```

_В конфигурационном файле `docker-compose.yml` параметр `subnet` должен совпадать с параметрами `redis_host` и `mongo_host` в конфигурационном файле приложения (`configs/example.yml`). Также, необходимо отредактировать `configs/alertmanager.yml` и `configs/prometheus.yml`, чтобы можно быть увидеть ТГ бота (`gateway` и `port`), который шлет алерты и правильно указать `target` для отслеживания прометеусом._

# Запуск приложения в kubernetes

## 1. Установка `cubectl`, `minikube` (если не установлены другие инструменты)

```
https://kubernetes.io/ru/docs/tasks/tools/
```

## 2. Запуск minikube

Для того, чтобы сделать shared Storage с датасетами и обученными моделями, необходимо примонтировать директорию к minikube

```
minikube start --mount=true --mount-string=$PWD/uploads:/tmp/uploads
```

## 3. Создание `configMap` (конфигурационного файла для приложений)

```
kubectl create configmap automl-config --from-file=configs dev.yml
```

В случае запуска в kubernetes необходимо в файле конфигурации (`configs/example.yml`) установить пареметры:
    - `redis_host` в значение `automl-redis-master`
    - `mongo_host` в значение `host.minikube.internal` (_Данная запись указывает на localhost, поскольку mongodb хостится не в kubernetes_)

## 4. Создание и запуск cущностей kubernetes

```
./start_kubernets.sh
```

## 5. Port forwarding

Чтобы сервис в `kubernetes` был виден "снаружи", необходимо пробросить порты

```
kubectl port-forward svc/automl-api 58080:8080
```

## 6. kubernetes cleanup

```
./delete_kubernetes.sh
```


## Документация API

Документация *Swagger* по ендпоинтам входным параметрам API располагается по адресу `http://localhost:58080/api/doc`