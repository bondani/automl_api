import logging
import pathlib

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

    logging.info('MongoDB is initialized')

    return client

def setup_logging(app, config):

    log_config = config['logging']
    
    log_filename = pathlib.Path(log_config.get('filename', 'app.log'))
    log_format = log_config.get(
        'format',
        '%(levelname)s::%(asctime)s::%(message)s',
        )
    log_debug = log_config.get('debug')

    if log_debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    if not log_filename.exists():
        log_filename.parent.mkdir(parents=True, exist_ok=True)
        log_filename.touch(exist_ok=True)


    logging.basicConfig(
        filename=log_filename,
        format=log_format,
        filemode='w',
        datefmt='%d-%b-%y %H:%M:%S',
        level=log_level
        )


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

    setup_logging(app, config)
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