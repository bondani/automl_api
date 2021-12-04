import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


def load_data(dataset_path):
    df = pd.read_csv(dataset_path)

    return df


def downcast(data):
    """Downcast dataframe (column-wise).

    Input
    ----------
    data (pandas.DataFrame)

    Returns
    -------
    Downcasted dataframe"""

    def downcast_column(data, column):
        result = pd.to_numeric(data[column], downcast='integer')
        if result.dtype == 'float64':
            result = pd.to_numeric(result, downcast='float')
        return result

    cols = [f for f, dtype in zip(data.columns, data.dtypes) if
            str(dtype).startswith('int') or str(dtype).startswith('float')]
    for f in cols:
        data[f] = downcast_column(data, f)
    return data


def feature_separation(data):
    '''Get 2 lists of names numerical and categorical features.

    Input
    ----------
    data (pandas.DataFrame)

    Returns
    -------
    list of numerical features
    list of categorical features'''
    num_feats, cat_feats = [], []
    for col in data.columns:
        mask_int = data[col].dtype == 'int8'
        mask_obj = data[col].dtype == 'object'
        mask_uniq = len(data[col].unique()) < 100
        if (mask_int & mask_uniq) | mask_obj:
            cat_feats.append(col)
        else:
            num_feats.append(col)
    return num_feats, cat_feats


def permutation_importance(df, y_true, model, n_steps=1, metric=roc_auc_score):
    '''Function for calculating Permutation importance.
    
    Parameters
    ----------
    df : data (pandas.DataFrame)
    y_true : Target values (array)
    model : fitted model with method predict
    n_steps : int, number of permutations of a column
    metric : function for calculating metric

    Returns
    -------
    Permutation importance : OrderedDict
    '''
    imp = OrderedDict() # ordered dict can be replaced with dict, list or array
    base_pred = model.predict(df)
    base_score = metric(y_true, base_pred)
    for col in df.columns:
        imp[col] = 0
        saved = df[col].values.copy() # saving unshuffled values of column
        for i in range(n_steps): 
            df[col] = np.random.permutation(df[col].values) # shuffling values of selected column
            pred = model.predict(df) # making prediction on dataset with shuffled column
            score = metric(y_true, pred) # calculating new score 
            diff = base_score-score # calculating importance of feature as difference between old and new score
            imp[col] += diff
            df[col] = saved
        imp[col] /= n_steps
    return imp
    
class AutoML():
    '''Auto-ML get different parameters of model and data.
    Fit different types of model:
    * for classification: 'lgbm', 'catboost', 'logreg'
    * for regression: 'lgbm', 'catboost', 'ridge'
    Returns
    -------
    Dictionary of chosen types fitted models
    -- Example:
    {'lgbm': {'model': lgbm_model,
              'selected_features': list of features},
     'logreg': {'model': logreg_model,
                'selected_features': list of features}}
    '''
    def __init__(self, model_type='classification', hyperparams={},\
                 selected_models='all', feature_selection=True):
        '''
        Parameters
        ----------
        * model_type: 'classification' or 'regression';
        * hyperparams: dict of dicts of hyperparameters:
        - for classification models: 'lgbm', 'logreg', 'catboost'
        - for regression models: 'ridge', 'lgbm', 'catboost'
        -- Classification example: 
           {'catboost': {'n_estimators':100},
            'logreg': {'penalty': 'l2'}}
        -- Regression example: 
           {'ridge': {'normalize': True,
                      'alpha': 0.8},
            'lgbm': {'depth': 5}};
        * selected_models: 'all' or list of selected models
        -- Example: ['ridge'];
        * feature_selection: True or False 
        Return top-30 permutation importance features
        (If n_features < 30 => False)
        '''
        self.model_type = model_type
        self.hyperparams = hyperparams
        if (model_type == 'classification') & (selected_models=='all'):
            self.selected_models = ['lgbm', 'logreg', 'catboost']
        elif (model_type == 'regression') & (selected_models=='all'):
            self.selected_models = ['ridge', 'lgbm', 'catboost']
        else:
            self.selected_models = selected_models
        self.feature_selection = feature_selection
        if model_type == 'classification':
            self.folds = StratifiedShuffleSplit(n_splits=5, train_size=0.8, random_state=42)
        elif model_type == 'regression':
            self.folds = ShuffleSplit(n_splits=5, train_size=0.8, random_state=42)

    def fit(self, X_train, y_train):
        self.models = {}

        X_train = downcast(X_train)
        self.num_feats, self.cat_feats = feature_separation(X_train)


        if self.model_type == 'classification':
            if 'lgbm' in self.selected_models:
                self.models['lgbm'] = self.lgbm_model(X_train, y_train)
            if 'logreg' in self.selected_models:
                self.models['logreg'] = self.logreg_model(X_train, y_train)
            if 'catboost' in self.selected_models:
                self.models['catboost'] = self.catboost_model(X_train, y_train)
        elif self.model_type == 'regression':
            if 'lgbm' in self.selected_models:
                self.models['lgbm'] = self.lgbm_model(X_train, y_train)
            if 'ridge' in self.selected_models:
                self.models['ridge'] = self.ridge_model(X_train, y_train)
            if 'catboost' in self.selected_models:
                self.models['catboost'] = self.catboost_model(X_train, y_train)

        return self.models

    def ridge_model(self, X_train, y_train):
        if self.feature_selection:
            perm_imp = np.zeros(X_train.shape[1])
            for k, (tr, te) in enumerate(self.folds.split(X_train, y_train)):
                xtr, xte = X_train.loc[tr], X_train.loc[te]
                ytr, yte = y_train[tr], y_train[te]
                if 'ridge' in self.hyperparams.keys():
                    model = Ridge(**self.hyperparams['ridge'])
                else:
                    model = Ridge()
                model.fit(xtr, ytr)
                perm_imp += list(permutation_importance(xte, yte, model, 1, r2_score).values())
            perm_imp /= 5
            all_imp = pd.DataFrame(index=X_train.columns)
            all_imp['permutation'] = perm_imp
            feats = all_imp.sort_values('permutation', ascending=False).index[:30]
        else:
            feats = X_train.columns
        if 'ridge' in self.hyperparams.keys():
            model = Ridge(**self.hyperparams['ridge'])
        else:
            model = Ridge()
        model.fit(X_train[feats], y_train)
        return {'model': model, \
                'selected_features': feats}

    def logreg_model(self, X_train, y_train):
        if self.feature_selection:
            perm_imp = np.zeros(X_train.shape[1])
            for k, (tr, te) in enumerate(self.folds.split(X_train, y_train)):
                xtr, xte = X_train.loc[tr], X_train.loc[te]
                ytr, yte = y_train[tr], y_train[te]
                if 'logreg' in self.hyperparams.keys():
                    model = LogisticRegression(**self.hyperparams['logreg'])
                else:
                    model = LogisticRegression()
                model.fit(xtr, ytr)
                perm_imp += list(permutation_importance(xte, yte, model, 1, accuracy_score).values())
            perm_imp /= 5
            all_imp = pd.DataFrame(index=X_train.columns)
            all_imp['permutation'] = perm_imp
            feats = all_imp.sort_values('permutation', ascending=False).index[:30]
        else:
            feats = X_train.columns
        if 'logreg' in self.hyperparams.keys():
            model = LogisticRegression(**self.hyperparams['logreg'])
        else:
            model = LogisticRegression()
        model.fit(X_train[feats], y_train)
        return {'model': model, \
                'selected_features': feats}

    def lgbm_model(self, X_train, y_train):
        if self.model_type == 'classification':
            if self.feature_selection:
                perm_imp = np.zeros(X_train.shape[1])
                for k, (tr, te) in enumerate(self.folds.split(X_train, y_train)):
                    xtr, xte = X_train.loc[tr], X_train.loc[te]
                    ytr, yte = y_train[tr], y_train[te]
                    if 'lgbm' in self.hyperparams.keys():
                        model = LGBMClassifier(**self.hyperparams['lgbm'])
                    else:
                        model = LGBMClassifier()
                    fit_params = {'eval_set': (xte, yte), \
                                  'early_stopping_rounds': 10,
                                  'verbose': False}
                    model.fit(xtr, ytr, **fit_params)
                    perm_imp += list(permutation_importance(xte, yte, model, 1, accuracy_score).values())
                perm_imp /= 5
                all_imp = pd.DataFrame(index=X_train.columns)
                all_imp['permutation'] = perm_imp
                feats = all_imp.sort_values('permutation', ascending=False).index[:30]
            else:
                feats = X_train.columns
            if 'lgbm' in self.hyperparams.keys():
                model = LGBMClassifier(**self.hyperparams['lgbm'])
            else:
                model = LGBMClassifier()
            model.fit(X_train[feats], y_train, verbose=False)
            return {'model': model, \
                    'selected_features': feats}
        else:
            if self.feature_selection:
                perm_imp = np.zeros(X_train.shape[1])
                for k, (tr, te) in enumerate(self.folds.split(X_train, y_train)):
                    xtr, xte = X_train.loc[tr], X_train.loc[te]
                    ytr, yte = y_train[tr], y_train[te]
                    if 'lgbm' in self.hyperparams.keys():
                        model = LGBMRegressor(**self.hyperparams['lgbm'])
                    else:
                        model = LGBMRegressor()
                    fit_params = {'eval_set': (xte, yte), \
                                  'early_stopping_rounds': 10,
                                  'verbose': False}
                    model.fit(xtr, ytr, **fit_params)
                    perm_imp += list(permutation_importance(xte, yte, model, 1, r2_score).values())
                perm_imp /= 5
                all_imp = pd.DataFrame(index=X_train.columns)
                all_imp['permutation'] = perm_imp
                feats = all_imp.sort_values('permutation', ascending=False).index[:30]
            else:
                feats = X_train.columns
            if 'lgbm' in self.hyperparams.keys():
                model = LGBMRegressor(**self.hyperparams['lgbm'])
            else:
                model = LGBMRegressor()
            model.fit(X_train[feats], y_train, verbose=False)
            return {'model': model, \
                    'selected_features': feats}

    def catboost_model(self, X_train, y_train):
        if self.model_type == 'classification':
            if self.feature_selection:
                perm_imp = np.zeros(X_train.shape[1])
                for k, (tr, te) in enumerate(self.folds.split(X_train, y_train)):
                    xtr, xte = X_train.loc[tr], X_train.loc[te]
                    ytr, yte = y_train[tr], y_train[te]
                    if 'catboost' in self.hyperparams.keys():
                        model = CatBoostClassifier(cat_features=self.cat_feats, \
                                                   verbose=False, \
                                                   early_stopping_rounds=10, \
                                                   **self.hyperparams['catboost'])
                    else:
                        model = CatBoostClassifier(cat_features=self.cat_feats, \
                                                   verbose=False, \
                                                   early_stopping_rounds=10)
                    model.fit(xtr, ytr)
                    perm_imp += list(permutation_importance(xte, yte, model, 1, accuracy_score).values())
                perm_imp /= 5
                all_imp = pd.DataFrame(index=X_train.columns)
                all_imp['permutation'] = perm_imp
                feats = all_imp.sort_values('permutation', ascending=False).index[:30]
            else:
                feats = X_train.columns
            new_cat_feats = list(set(self.cat_feats).intersection(set(feats)))
            if 'catboost' in self.hyperparams.keys():
                model = CatBoostClassifier(cat_features=new_cat_feats, \
                                           verbose=False, \
                                           **self.hyperparams['catboost'])
            else:
                model = CatBoostClassifier(cat_features=new_cat_feats, \
                                           verbose=False)
            model.fit(X_train[feats], y_train)
            return {'model': model, \
                    'selected_features': feats}
        else:
            if self.feature_selection:
                perm_imp = np.zeros(X_train.shape[1])
                for k, (tr, te) in enumerate(self.folds.split(X_train, y_train)):
                    xtr, xte = X_train.loc[tr], X_train.loc[te]
                    ytr, yte = y_train[tr], y_train[te]
                    if 'catboost' in self.hyperparams.keys():
                        model = CatBoostRegressor(cat_features=self.cat_feats, \
                                                  verbose=False, \
                                                  early_stopping_rounds=10, \
                                                  **self.hyperparams['catboost'])
                    else:
                        model = CatBoostRegressor(cat_features=self.cat_feats, \
                                                  verbose=False, \
                                                  early_stopping_rounds=10)
                    model.fit(xtr, ytr)
                    perm_imp += list(permutation_importance(xte, yte, model, 1, r2_score).values())
                perm_imp /= 5
                all_imp = pd.DataFrame(index=X_train.columns)
                all_imp['permutation'] = perm_imp
                feats = all_imp.sort_values('permutation', ascending=False).index[:30]
            else:
                feats = X_train.columns
            new_cat_feats = list(set(self.cat_feats).intersection(set(feats)))
            if 'catboost' in self.hyperparams.keys():
                model = CatBoostRegressor(cat_features=new_cat_feats, \
                                          verbose=False, \
                                          **self.hyperparams['catboost'])
            else:
                model = CatBoostRegressor(cat_features=new_cat_feats, \
                                          verbose=False)
            model.fit(X_train[feats], y_train)
            return {'model': model, \
                    'selected_features': feats}

def predict(X_test, models):
    '''Predict values of data by fitted models.
    Parameters
    ----------
    X_test: data (pd.DataFrame)
    models: dictionary of fitted models
    -- Example:
    {'lgbm': {'model': lgbm_model,
              'selected_features': list of features},
     'logreg': {'model': logreg_model,
                'selected_features': list of features}}
    Returns
    ----------
    Dictionary of predictions of lists np.ndarray
    -- Example:
    {'lgbm': np.ndarray([0, 1, 0,..., 0]),
     'catboost': np.ndarray([1, 1, 0,..., 0])}
    '''
    predictions = {}
    X_test = downcast(X_test)
    
    if 'lgbm' in models.keys():
        feats = models['lgbm']['selected_features']
        predictions['lgbm'] = models['lgbm']['model'].predict(X_test[feats]).tolist()
    if 'logreg' in models.keys():
        feats = models['logreg']['selected_features']
        predictions['logreg'] = models['logreg']['model'].predict(X_test[feats]).tolist()
    if 'ridge' in models.keys():
        feats = models['ridge']['selected_features']
        predictions['ridge'] = models['ridge']['model'].predict(X_test[feats]).tolist()
    if 'catboost' in models.keys():
        feats = models['catboost']['selected_features']
        predictions['catboost'] = models['catboost']['model'].predict(X_test[feats]).tolist()
    
    return predictions

def predict_proba(X_test, models):
    '''Predict probability values of data by fitted models
    for classification models.
    Parameters
    ----------
    X_test: data (pd.DataFrame)
    models: dictionary of fitted models
    -- Example:
    {'lgbm': {'model': lgbm_model,
              'selected_features': list of features},
     'logreg': {'model': logreg_model,
                'selected_features': list of features}}
    Returns
    ----------
    Dictionary of probability predictions (lists np.ndarray)
    -- Example:
    {'lgbm': np.ndarray([[0.21, 0.99,..., 0.23], [0.79, 0.01,..., 0.77]]),
     'catboost': np.ndarray([[0.35, 0.8,..., 0.2], [0.65, 0.2,..., 0.8]])}
    '''
    predictions = {}
    X_test = downcast(X_test)

    if 'lgbm' in models.keys():
        feats = models['lgbm']['selected_features']
        predictions['lgbm'] = models['lgbm']['model'].predict_proba(X_test[feats]).tolist()
    if 'logreg' in models.keys():
        feats = models['logreg']['selected_features']
        predictions['logreg'] = models['logreg']['model'].predict_proba(X_test[feats]).tolist()
    if 'catboost' in models.keys():
        feats = models['catboost']['selected_features']
        predictions['catboost'] = models['catboost']['model'].predict_proba(X_test[feats]).tolist()
    
    return predictions