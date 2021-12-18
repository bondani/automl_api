import csv
from pathlib import Path

from aiohttp import web

import aiofiles

async def get_datasets(request):

    """
    ---
    description: This end-point response all loaded datasets.
    produces:
    - application/json
    responses:
        "200":
            description: successful operation. Return JSON object with datasets_list array
    """

    datasets_path = Path(request.app['config']['storage']['datasets'])

    files = [filename.name for filename in datasets_path.glob('**/*') if filename.is_file()]

    resp = {
        'datasets_files': files 
    }

    return web.json_response(resp)


async def add_dataset(request):

    """
    ---
    description: This end-point allow to add new dataset file.
    consumes:
    - application/json
    produces:
    - application/json
    parametes:
    - name: filename
      in: path
      description: Dataset name with suffix .csv; Dataset should contain \"target\" field
      required: true
      type: string
    responses:
        "400":
            description: Error. Bad filename
        "200":
            description: successful operation. Return JSON object with added dataset name
    """

    datasets_path = Path(request.app['config']['storage']['datasets'])

    filename = Path(request.match_info['filename'])
    
    if filename.suffix != '.csv':
        filename = filename.with_stem('.csv')

    file_path = datasets_path / filename

    if file_path.is_file():
        return web.json_response({
            'status': 'error',
            'msg': f'file {filename} already exists'
        })

    async for data in request.content.iter_chunked(1024):

        if file_path.is_file():
            file_mode = 'ba'
        
        else:
            file_mode = 'wb'

        async with aiofiles.open(file_path, file_mode) as f:
            await f.write(data)

    logging.info(f'{filename.name} dataset add')

    return web.json_response({
        'status': 'success',
        'added_dataset': filename.name
        })
    

    pass