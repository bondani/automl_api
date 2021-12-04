import uuid
from pathlib import Path

import pandas as pd
import pymongo


def get_db(db_host, db_port, db_name):
    client = pymongo.MongoClient(db_host, db_port)
    db = client[db_name]

    return db


def read_dataset(dataset_path):
    p = Path(dataset_path)

    df = pd.read_csv(p)

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    return df


def dump_models(models, dataset, config):

    print('models', models, type(models))

    import pickle as pkl

    filename = str(uuid.uuid4()) + '.pkl'

    model_path = Path(config['storage']['models']) / filename

    with open(model_path, 'wb') as f:
        pkl.dump(models, f)

    for model in models:
        models[model].pop('model')
        models[model]['selected_features'] = list(models[model]['selected_features'])

    models['dataset'] = dataset

    models['filename'] = filename

    db_host = config['mongodb']['mongo_host']
    db_port = config['mongodb']['mongo_port']
    db_name = config['mongodb']['mongo_db_name']

    db = get_db(db_host, db_port, db_name)

    available_models = db['available_models']
    available_models.insert_one(models)


    

    


