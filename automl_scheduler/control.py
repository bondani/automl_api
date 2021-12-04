import re
import time
from pathlib import Path


from redis import Redis
from rq import Queue
from rq.registry import StartedJobRegistry
from rq.job import Job

from automl_scheduler.automl import AutoML
from automl_scheduler.automl_helper import read_dataset
from automl_scheduler.automl_helper import dump_models


def on_success(job, connection, result, *args, **kwargs):

    job.refresh()
    dataset = job.meta['dataset']
    config = job.meta['config']

    dump_models(job.result, dataset, config)


class Scheduler:

    def __init__(self, config):
        self.config = config

        self.models_path = Path(config['storage']['models'])

        redist_host = config['rq']['redis_host']
        redist_port = config['rq']['redis_port']

        self.r = Redis(host=redist_host, port=redist_port)
        self.q = Queue(connection=self.r)


    def enqueue_job(self,
                    dataset,
                    model_type,
                    hyperparams,
                    selected_models,
                    feature_selection):

        automl = AutoML(model_type=model_type, hyperparams=hyperparams, 
                    selected_models=selected_models, feature_selection=feature_selection)
        
        df = read_dataset(dataset)
        
        job = self.q.enqueue(
            automl.fit,
            df.drop('target', axis=1),
            df['target'],
            on_success=on_success,
            )
        
        job.meta['dataset'] = dataset.name
        job.meta['model_type'] = model_type
        job.meta['config'] = self.config
        job.save()


            
