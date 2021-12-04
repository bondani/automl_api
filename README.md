# Запуск веб-приложения

## 1. Запустить инстанс БД Mongo

```
docker run --name some-mongo -p 57017:27017 mongo
```

## 2. Запустить инстанс Redis

```
docker run --name some-redis -p 56379:6379 --rm -d redis
```

## 3. Создать образ докера веб-приложения

```
docker build -t automl_api .
```

## 4. Запусть контейнер на основе созданного образа

```
docker run -d -p 58080:8080 -v ${PWD}/uploads:/backend/uploads --name some-automl automl_api
```

## Документация API

Документация *Swagger* по ендпоинтам входным параметрам API располагается по адресу `http://localhost:58080/api/doc`