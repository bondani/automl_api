import logging

from aiohttp import web
import motor.motor_asyncio as aiomotor
from aiohttp_swagger import setup_swagger
import prometheus_client
from prometheus_client import Counter

from automl_api.routes import setup_routes
from automl_api.settings import load_config
from automl_api.middlewares import prometheus_middleware
from automl_api.helpers.prometheus import setup_prometheus

from automl_scheduler.control import Scheduler


async def init_mongo(config):
    mongo_host = config['mongodb']['mongo_host']
    mongo_port = config['mongodb']['mongo_port']

    client = aiomotor.AsyncIOMotorClient(mongo_host, mongo_port)

    return client


def setup_middlewares(app):

    app_name = app['app_name']

    middlewares = [
        prometheus_middleware(app_name)
    ]

    app.middlewares.extend(middlewares)


async def init_app(config):
    app = web.Application()

    app['config'] = config
    app['app_name'] = config['app'].get('app_name', 'not_setup')

    setup_routes(app)
    setup_swagger(app)

    scheduler = Scheduler(app['config'])
    app['scheduler'] = scheduler

    db = await init_mongo(app['config'])
    app['db'] = db

    setup_prometheus(app)

    setup_middlewares(app)

    return app


def main(configpath):
    config = load_config(configpath)
    app = init_app(config)
    web.run_app(app)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    
    args = parser.parse_args()

    if args.config:
        main(args.config)

    else:
        parser.print_help()