from pathlib import Path
import json

from aiohttp import web
from bson import json_util
from bson.objectid import ObjectId
from aiohttp_swagger import swagger_path

from automl_scheduler.automl import predict, predict_proba
from automl_api.automl_helper import validate_automl_params, load_model
from automl_scheduler.automl_helper import read_dataset


async def get_available_models_classes(request):
    """
    ---
    description: This end-point response all available models classes.
    produces:
    - "application/json"
    responses:
        "200":
            description: successful operation. Return JSON object with available models classes
    """
    config = request.app['config']
    amc = config['automl']['available_models_classes']
    
    return web.json_response(amc)


async def get_models(request):
    """
    ---
    description: This end-point response all available trained models.
    produces:
    - "application/json"
    responses:
        "200":
            description: successful operation. Return JSON object with available trained models
    """
    db_name = request.app['config']['mongodb']['mongo_db_name']
    db = request.app['db'][db_name]

    models = await db['available_models'].find().to_list(length=None)

    return web.json_response(json.loads(json_util.dumps(models)))


async def remove_model(request):
    """
    ---
    description: This end-point allow to delete trained model.
    produces:
    - application/json
    parameters:
    - in: body
      name: body
      description: Model parameters object
      required: true
      schema:
        type: object
        properties:
          model_id:
            type: string
            description: model id from page GET /api/models
    responses:
        "400":
            description: Error. Bad model id
        "200":
            description: successful operation. Remove model and model record and return removed model
    """
   
    data = await request.json()
    print(data)
    _id = data.get('model_id')

    if not _id:
        return {
            'status': 'error',
            'msg': 'model_id need to be specify'
        }

    db_name = request.app['config']['mongodb']['mongo_db_name']
    db = request.app['db'][db_name]

    model = await db['available_models'].find_one_and_delete({"_id": ObjectId(_id)})

    if not model:
        return web.json_response({
            'status': 'error',
            'msg': f'model_id: {_id} not found'
        })

    model = json.loads(json_util.dumps(model))

    filename = model.get('filename')

    model_path = Path(request.app['config']['storage']['models']) / filename

    try:
        model_path.unlink()
    except FileNotFoundError:
        pass

    return web.json_response({
        'status': 'success',
        'result': model
    })


async def fit_model(request):
    """
    ---
    description: This end-point allow to select model, specify model parametrs and fit.
    consumes:
    - application/json
    produces:
    - application/json
    parameters:
    - in: body
      name: body
      description: Model parameters object
      required: true
      schema:
        type: object
        properties:
          dataset:
            type: string
            required: true
            description: Dataset name with suffix .csv, that available on GET /api/datasets
          model_type:
            type: string
            enum:
            - classification
            - regression
            default: classification
          hyperparams:
            type: object
            required: false
            default: {}
            properties:
              lgbm:
                type: object
                description: enumeration hyperparams lightgbm model
              catboost:
                 type: object
                 description: enumeration hyperparams catboost model
              ridge:
                 type: object
                 description: enumeration hyperparams ridge model
              logreg:
                 type: object
                 description: enumeration hyperparams logreg model
          selected_models:
            type:
              - string
              - array
            required: false
            default: all
            description: enumeration models to train
          feature_selection:
            type: boolean
            default: true
            description: choose top-30 features sorted by permutation importance   
    responses:
        "400":
            description: Error. Bad params
        "200":
            description: successful operation. Add fit job to queue
    """

    data = await request.json()

    dataset = data.get('dataset')

    if not dataset:
        return web.json_response({
            'status': 'Error',
            'msg': 'Need to specify dataset'
        })

    datasets_path = Path(request.app['config']['storage']['datasets'])

    dataset_path = datasets_path / dataset

    if not dataset_path.is_file():
        return web.json_response({
            'status': 'Error',
            'msg': 'Dataset not found'
        })

    model_type = data.get('model_type', 'classification')
    hyperparams = data.get('hyperparams', {})
    selected_models = data.get('selected_models', 'all')
    feature_selection = data.get('feature_selection', True)

    val_res = validate_automl_params(
        dataset_path,
        model_type,
        hyperparams,
        selected_models,
        feature_selection
    )

    if val_res:
        return web.json_response(val_res)


    request.app['scheduler'].enqueue_job(
        dataset_path,
        model_type,
        hyperparams,
        selected_models,
        feature_selection
        )


    return web.json_response({
        'status': 'success',
        'msg': 'Job in queue'
    })


async def predict_data(request):
    """
    ---
    description: Make predict on dataset with availiable trained models
    consumes:
    - application/json
    produces:
    - application/json
    parameters:
    - in: body
      name: body
      description: Model parameters object
      required: true
      schema:
        type: object
        properties:
          dataset:
            type: string
            required: true
            description: Dataset name with suffix .csv, that available on GET /api/datasets
          model_id:
            type: string
            required: true
            description: id of trained model. Can be found in GET /api/models
          proba:
            type: boolean
            default: false
            description: To predict probas in classification models
    responses:
        "400":
            description: Error. Bad params
        "200":
            description: successful operation. Predict result for all selected autoML models
    """
    data = await request.json()

    _id = data.get('model_id')
    dataset = data.get('dataset')
    proba = data.get('proba', False)

    if not dataset:
        return {
            'status': 'error',
            'msg': 'dataset need to be specify'
        }

    if not _id:
        return {
            'status': 'error',
            'msg': 'model_id need to be specify'
        }

    datasets_path = Path(request.app['config']['storage']['datasets'])
    dataset_path = datasets_path / dataset

    if not dataset_path.is_file():
        return web.json_response({
            'status': 'Error',
            'msg': 'Dataset not found'
        })

    db_name = request.app['config']['mongodb']['mongo_db_name']
    db = request.app['db'][db_name]

    model = await db['available_models'].find({"_id": ObjectId(_id)}).to_list(length=1)

    if not model:
        return {
            'status': 'error',
            'msg': f'model_id: {_id} not found'
        }

    model = json.loads(json_util.dumps(model))[0]

    filename = model.get('filename')

    model_path = Path(request.app['config']['storage']['models']) / filename

    df = read_dataset(dataset_path)
    model = load_model(model_path)

    if proba:
        try:
            res = predict_proba(df, model)

        except AttributeError:
            return web.json_response({
                'status': 'error',
                'msg': 'predict proba not available for non classification model'
            })

        if not res:
            return web.json_response({
                'status': 'error',
                'msg': 'predict proba not available for non classification model'
            })
    
    else:
        res = predict(df, model)

    return web.json_response(res)