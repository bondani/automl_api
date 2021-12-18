from automl_api.views.models import get_models, fit_model, predict_data, remove_model
from automl_api.views.models import get_available_models_classes
from automl_api.views.datasets import get_datasets, add_dataset
from automl_api.views.prometheus import metrics

def setup_routes(app):
    app.router.add_get('/api/models_classes', get_available_models_classes, name='get_available_models_classes')
    app.router.add_get('/api/models', get_models, name='get_models')
    app.router.add_post('/api/models', fit_model, name='fit_model')
    app.router.add_delete('/api/models', remove_model, name='remove_model')
    app.router.add_post('/api/models/predict', predict_data, name='predict_data')
    app.router.add_get('/api/datasets', get_datasets, name='get_datasets')
    app.router.add_post('/api/datasets/{filename}', add_dataset, name='add_dataset')
    app.router.add_get('/metrics', metrics, name='metrics')