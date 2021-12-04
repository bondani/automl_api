import pandas as pd


def load_model(model_path):
    import pickle as pkl

    print(model_path)

    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    return model


def validate_automl_params(
    dataset_path,
    model_type,
    hyperparams,
    selected_models,
    feature_selection):

    df = pd.read_csv(dataset_path)

    if 'target' not in df.columns:
        return {
            'status': 'error',
            'msg': '"target" not in dataset'
        }

    if model_type not in ['classification', 'regression']:
        return {
            'status': 'error',
            'msg': 'model_type should be classification or regression'
        }

    if model_type == 'classification':
        if len(df['target'].unique()) == len(df):
            return {
                'status': 'error',
                'msg': 'Number of unqiue values equal dataset length'
            }

    if not isinstance(hyperparams, dict):
        return {
            'status': 'error',
            'msg': 'hyperparams should be type dict'
        }

    if not isinstance(feature_selection, bool):
        return {
            'status': 'error',
            'msg': 'feature_selection should be type bool'
        }

    if model_type == 'classification':
        if not set(hyperparams.keys()).issubset(set(['lgbm', 'catboost', 'logreg'])):
            return {
                'status': 'error',
                'msg': 'Keys of hyperparams classification model can be "lgbm", "catboost", "logreg"'
            }

    if model_type == 'regression':
        if not set(hyperparams.keys()).issubset(set(['lgbm', 'catboost', 'ridge'])):
            return {
                'status': 'error',
                'msg': 'Keys of hyperparams regression model can be "lgbm", "catboost", "ridge"'
            }

    if model_type == 'classification':
        
        if selected_models != 'all' and not set(selected_models).issubset(set(['lgbm', 'catboost', 'logreg'])):
            return {
                'status': 'error',
                'msg': 'selected_models of classification model can be list of "lgbm", "catboost", "logreg" or string "all"'
            }

    if model_type == 'regression':
        if selected_models != 'all' and not set(selected_models).issubset(set(['lgbm', 'catboost', 'ridge'])):
            return {
                'status': 'error',
                'msg': 'selected_models of regression model can be list of "lgbm", "catboost", "ridge" or string "all"'
            }
    
