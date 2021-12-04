import logging

from aiohttp import web
import motor.motor_asyncio as aiomotor
from aiohttp_swagger import setup_swagger

from automl_api.routes import setup_routes
from automl_api.settings import load_config

from automl_scheduler.control import Scheduler


async def init_mongo(config):
    mongo_host = config['mongodb']['mongo_host']
    mongo_port = config['mongodb']['mongo_port']

    client = aiomotor.AsyncIOMotorClient(mongo_host, mongo_port)

    return client


async def init_app(config):
    app = web.Application()

    app['config'] = config

    setup_routes(app)
    setup_swagger(app)

    scheduler = Scheduler(app['config'])
    app['scheduler'] = scheduler

    db = await init_mongo(app['config'])
    app['db'] = db

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