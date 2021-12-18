import pytest
import pathlib

from automl_api.__main__ import init_app
from automl_api.settings import load_config


@pytest.fixture
def config():
    dev_config = pathlib.Path('configs') / 'example.yml'
    config = load_config(dev_config)

    return config

@pytest.fixture
async def client(aiohttp_client, config):
    app = await init_app(config)

    return await aiohttp_client(app)