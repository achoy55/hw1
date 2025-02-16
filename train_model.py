import pandas as pd
import numpy as np
from enum import Enum
from numpy import hstack
import joblib 
import os
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
tqdm.pandas()

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,TimeSeriesSplit, RandomizedSearchCV, GridSearchCV 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import import_ipynb
from data_validation import detect_SeasonalAD, detect_InterQuartileRangeAD, detect_AutoregressionAD, detect_QuantileAD
from data_validation import normalize_with_zcore, risk_rating_z_between_column, plot_anomalies, plot_anomalies_by_column, detect_anomalies_z, merge_anomalies_z
from data_validation import plot_anomalies, plot_anomalies_by_column
from data_validation import detect_anomalies_z, merge_anomalies_z
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

def save_model(model, name, ):
    os.makedirs('trained_model', exist_ok=True) 
    joblib.dump(model, f'trained_model/{name}.joblib')

def load_model(name):
    return joblib.load(f'trained_model/{name}.joblib')

def detect_anomalies_adtk(df, column_name):
    anomalies = detect_SeasonalAD(df[column_name], 3.0, 'both')
    print('Anomalies detected:', anomalies.value_counts())
    return anomalies

def linear_regression_model(X_train, y_train, params):
    model_params = params.copy()
    model = LinearRegression().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def logistic_regression_models(X_train, y_train, params):
    model_params = params.copy()
    models = list()
    for param in model_params:
        model = LogisticRegression().set_params(**param)
        model.fit(X_train, y_train)
        models.append(model)
    
    return models

def logistic_regression_model(X_train, y_train, params):
    """
    Logistic Regression: Топ 3 параметра для тюнинга
        - **solver**: Метод оптимизации. Популярные значения:
        - 'liblinear' (по умолчанию для небольших датасетов)
        - 'lbfgs' (подходит для многоклассовых задач)
        - 'saga' (подходит для больших датасетов и разреженных данных)
        - **C**: Инверсия регуляризации. Меньшие значения C увеличивают регуляризацию, что помогает избежать переобучения. Например, `C=1.0` (по умолчанию), `C=0.01` — более сильная регуляризация.
        - **max_iter**: Максимальное количество итераций для оптимизации. Увеличьте, если модель не сходится. Например, `max_iter=1000`.
    """
    model_params = params.copy()
    model = LogisticRegression().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def catboot_classifier_model(X_train, y_train, params):
    model_params = params.copy()
    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def catboot_regressor_model(X_train, y_train, params):
    model_params = params.copy()
    model = CatBoostRegressor(**model_params)
    model.fit(X_train, y_train)
    return model

def catboot_classifier_model_grid_search(X_train, y_train, X_val, y_val, parameters):
    grid_params = parameters['grid_params'].copy()
    params = parameters['params'].copy()
    model = CatBoostClassifier(random_state=params['random_state'], verbose=params['verbose'],
                               early_stopping_rounds=params['early_stopping_rounds'])
    grid_search = GridSearchCV(estimator=model, param_grid=grid_params, scoring=params['scoring'],
                               cv=params['cv'], n_jobs=params['n_jobs'])
    grid_search.fit(X_train, y_train, eval_set=(X_val, y_val))
    return grid_search

def catboot_regressor_model_grid_search(X_train, y_train, X_val, y_val, parameters):
    grid_params = parameters['grid_params'].copy()
    params = parameters['params'].copy()
    model = CatBoostRegressor(random_state=params['random_state'], verbose=params['verbose'],
                               early_stopping_rounds=params['early_stopping_rounds'], task_type=params['task_type'])
    grid_search = GridSearchCV(estimator=model, param_grid=grid_params, scoring=params['scoring'],
                               cv=params['cv'], n_jobs=params['n_jobs'])
    grid_search.fit(X_train, y_train, eval_set=(X_val, y_val))
    return grid_search

def random_forest_classifier_model(X_train, y_train, params):
    model_params = params.copy()
    model = RandomForestClassifier().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def random_forest_regressor_model(X_train, y_train, params):
    model_params = params.copy()
    model = RandomForestRegressor().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def decision_tree_classifier_model(X_train, y_train, params):
    model_params = params.copy()
    model = DecisionTreeClassifier().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def decision_tree_regressor_model(X_train, y_train, params):
    model_params = params.copy()
    model = DecisionTreeRegressor().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def decision_tree_classifier_model_grid_search(X_train, y_train, X_val, y_val, params):
    model_params = params.copy()
    model = DecisionTreeClassifier(random_state=model_params['random_state'],)
    grid_search = GridSearchCV(model, model_params, cv=model_params['cv'], scoring=model_params['scoring'], )
    # grid_search.fit(X_train, y_train)
    grid_search.fit(X_train, y_train, eval_set=(X_val, y_val))
    return grid_search

def decision_tree_regressor_model_grid_search(X_train, y_train, params):
    model_params = params.copy()
    model = DecisionTreeRegressor(random_state=model_params['random_state'],)
    grid_search = GridSearchCV(model, model_params, cv=model_params['cv'], scoring=model_params['scoring'], )
    grid_search.fit(X_train, y_train)
    return grid_search

def knn_classifier_model(X_train, y_train, params):
    model_params = params.copy()
    model = KNeighborsClassifier().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def knn_regressor_model(X_train, y_train, params):
    model_params = params.copy()
    model = KNeighborsRegressor().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def xgboost_classifier_model(X_train, y_train, params):
    model_params = params.copy()
    model = XGBClassifier().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def xgboost_regressor_model(X_train, y_train, params):
    model_params = params.copy()
    model = XGBRegressor().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def svc_classifier_model(X_train, y_train, params):
    model_params = params.copy()
    model = SVC().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

def lstm_model(X_train, y_train, params):
    n_steps = params['n_steps']
    n_features = params['n_features']
    batch_size = params['batch_size']
    epochs = params['epochs']
    units = params['units']
    lose = params['lose']
    optimizer = params['optimizer']
    validation_split = params['verbose']
    return_sequences = params['return_sequences']
    shuffle = params['shuffle']
    activation = params['activation']
    verbose = params['verbose']
    
    model = Sequential()
    # model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(units, activation=activation, return_sequences=return_sequences, input_shape=(n_steps, n_features)))
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss=lose)
    # model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)
    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, \
              shuffle=shuffle, validation_split=validation_split, verbose=verbose)
    return model

def model_fit(model_func, X_train, y_train, params):
    return model_func(X_train, y_train, params)

def model_fit_with_eval(model_func, X_train, y_train, eval_set, params):
    model_params = params.copy()
    if ModelFunc.CATBOOST_CLASS is model_func:
        model = CatBoostClassifier(**model_params, random_seed=42, verbose=0)
    if ModelFunc.CATBOOST_REG is model_func:
        model = CatBoostRegressor(**model_params, random_seed=42, verbose=0)
    model.fit(X_train, y_train, eval_set=eval_set)
    return model

def model_fit_with_eval_set(model_func, X_train, y_train, eval_set, params):
    return model_func(X_train, y_train, eval_set=eval_set, params=params)

def normalize_MinMaxScaler(X_train, X_val, X_test):
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def normalize_MinMax_by_column(X_train, y_train):
    sc = MinMaxScaler()
    X_df, y_df = sc.fit_transform(np.column_stack((X_train, y_train))).T
    return X_df, y_df

def normalize_StandardScaler(X_train, X_val, X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def standard_scaler(data):
    scaler = StandardScaler()
    data.loc[:] = scaler.fit_transform(data)
    return data

def roc_auc_score_metric(sample1, sample2):
    return roc_auc_score(sample1, sample2)

def calculate_metrics_table(y_true, y_pred_prob, thresholds=[0.5, 0.6, 0.7, 0.8]):
    metrics_table = []
    
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        metrics = {
            'Cutoff': threshold,
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            # 'ROC-AUC': roc_auc_score(y_true, y_pred)
        }
        metrics_table.append(metrics)
    
    return pd.DataFrame(metrics_table)*100

def display_sample_metrics(name, y_true, y_pred_prob):
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    print(f"[{name}] sample metrics: ROC AUC: {roc_auc:.4f}")
    print(calculate_metrics_table(y_true, y_pred_prob))

def top_n_weighted_factors(importance_function, features, top):
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance_function
    })
    
    top_features = feature_importance.reindex(feature_importance['Importance'].abs().sort_values(ascending=False).index).head(top)
    print(f"=== Top-{top} most important factors ===")
    print(top_features)

def fit_models(model_funcs, X_train, y_train, X_val=None, y_val=None):
    models = list()
    for model_func in model_funcs:
        params = get_model_params(model_func)
        if X_val is None or y_val is None:
            model = model_fit(model_func, X_train, y_train, params)
        else:
            model = model_fit_with_eval(model_func, X_train, y_train, eval_set=(X_val, y_val), params=params)

        models.append(model)

    return models

def predict_models(models, X_train, X_val, X_test):
    models_proba = list()
    for model in models:
        try:
            train_arr = np.array(model.predict_proba(X_train)[:, 1])
            val_arr = np.array(model.predict_proba(X_val)[:, 1])
            test_arr = np.array(model.predict_proba(X_test)[:, 1])
        except Exception as e:
            train_arr = np.array(model.predict(X_train))
            val_arr = np.array(model.predict(X_val))
            test_arr = np.array(model.predict(X_test))
        
        models_proba.append({
            'train': train_arr,
            'val': val_arr,
            'test': test_arr,
        })

    return models_proba

def blending_pred(*args):
    return sum(args) / len(args)
    
def stacking_pred(*args):
    return np.column_stack(args)

def show_importance(model, model_func, params):
    importance_function = model.coef_[0]
    if model_func in [ModelFunc.XGBOOST_CLASS, ModelFunc.DECISION_TREE_CLASS, ModelFunc.RANDOM_FOREST_CLASS, \
                      ModelFunc.KNN_CLASS, ]:
        importance_function = model.feature_importances_
    if model_func in [ModelFunc.CATBOOST_CLASS, ]:
        importance_function = model.get_feature_importance()

    top_n_weighted_factors(importance_function, params['features'], params['top'])
    return importance_function
    
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    use_groups = "Group" in type(cv).__name__
    groups = group if use_groups else None
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax

def split_data(data, params):
    df = data.copy()

    max_train_size = params['max_train_size']
    test_size = params['test_size']
    n_splits = int((len(df) - max_train_size) // test_size)

    # print(f'{n_splits}, {max_train_size}, {test_size}')

    tss = TimeSeriesSplit(n_splits = n_splits, max_train_size=max_train_size, test_size=test_size)
    for train_index, test_index in tss.split(df.index):
        X_train, y_train = df.iloc[train_index,:], df.iloc[train_index]
        test_data = df.iloc[test_index]
        train_data, val_data, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size/100, shuffle=False)
        yield (
            train_data, val_data, test_data 
        )


def split_data_by_date(data_with_features):
    test_start_date = pd.to_datetime(data_with_features.index.max()) - pd.DateOffset(months=1)
    val_start_date = pd.to_datetime(data_with_features.index.max()) - pd.DateOffset(months=2)

    train_data = data_with_features[pd.to_datetime(data_with_features.index) < val_start_date] 
    val_data = data_with_features[(pd.to_datetime(data_with_features.index) >= val_start_date) & \
                                  (pd.to_datetime(data_with_features.index) < test_start_date)]
    test_data = data_with_features[pd.to_datetime(data_with_features.index) >= test_start_date]
    # print(data_with_features.index[-1], train_data.index[-1], val_data.index[-1], test_data.index[-1])    

    return train_data, val_data, test_data

def split_data_by_date2(data, params):
    # Определяем дату начала тестовой выборки (последний месяц)
    test_start_date = data.index.max() - pd.DateOffset(months=params['last_test_month_cnt'])

    # Определяем дату начала валидационной выборки (предпоследний месяц)
    val_start_date = data.index.max() - pd.DateOffset(months=params['last_val_month_cnt'])

    # Разделение данных на тренировочную, валидационную и тестовую выборки по времени
    train_data = data[data.index < val_start_date]  # все, что до предпоследнего месяца
    val_data = data[(data.index >= val_start_date) & (data.index < test_start_date)]  # предпоследний месяц
    test_data = data[data.index >= test_start_date]  # последний месяц

    return train_data, val_data, test_data

def split_by_features_and_target_variables(data, features):
    """ Разделение на признаки (X) и целевую переменную (y) для каждой выборки """
    X_data = data[features]
    y_data = data['Target']
    return X_data, y_data

def predict_ensemble(data_with_features, params):
    model_funcs = params['model_funcs']
    features = params['features']

    print(f'=== Start Train models:\n {model_funcs} ===')

    for train_data, val_data, test_data in split_data(data_with_features, params):
        X_train, y_train = train_data[features], train_data['Target']
        X_val, y_val     = val_data[features], val_data['Target']
        X_test, y_test   = test_data[features], test_data['Target']

        ## Data normalization
        # X_train_scaled, X_val_scaled, X_test_scaled = ut.normalize_MinMaxScaler(X_train, X_val, X_test)
        X_train_scaled, X_val_scaled, X_test_scaled = normalize_StandardScaler(X_train, X_val, X_test)
    
        ## Modeling
        # models = fit_models(model_funcs, X_train_scaled, y_train, X_val=X_val, y_val=y_val)
        models = fit_models(model_funcs, X_train_scaled, y_train)

        ## Prediction on train, val and test samples
        predict_dict = predict_models(models, X_train_scaled, X_val_scaled, X_test_scaled)

        yield ( predict_dict, y_train, y_val, y_test )


def predict_process(data_with_features, params):
    use_stacking = params['use_stacking']
    use_blending= params['use_blending']
    
    ## Split, predict
    y_train_pred_prob = list()
    y_val_pred_prob = list()
    y_test_pred_prob = list()
    y_train_total = pd.DataFrame()
    y_val_total = pd.DataFrame()
    y_test_total = pd.DataFrame()

    for predict_dict, y_train, y_val, y_test in predict_ensemble(data_with_features, params):
        y_train_total = pd.concat([y_train_total, y_train], ignore_index=True)
        y_val_total = pd.concat([y_val_total, y_val], ignore_index=True)
        y_test_total = pd.concat([y_test_total, y_test], ignore_index=True)

        y_train_pred_prob.append([d['train'] for d in predict_dict][0])
        y_val_pred_prob.append([d['val'] for d in predict_dict][0])
        y_test_pred_prob.append([d['test'] for d in predict_dict][0])

    ## 2D-array
    train_pred_prob = hstack(y_train_pred_prob)
    val_pred_prob = hstack(y_val_pred_prob)
    test_pred_prob = hstack(y_test_pred_prob)

    print(f"     Train size: {len(y_train_total)}, Val size: {len(y_val_total)}, Test size: {len(y_test_total)}")
    print(f"Pred Train size: {len(train_pred_prob)}, Val size: {len(val_pred_prob)}, Test size: {len(test_pred_prob)}")

    ## Final model using whole data
    if use_stacking:
        stacked_train_X = stacking_pred(train_pred_prob).reshape(-1,1)
        stacked_val_X = stacking_pred(val_pred_prob).reshape(-1,1)
        stacked_test_X = stacking_pred(test_pred_prob).reshape(-1,1)
        
        final_model_func = ModelFunc.LOGISTIC_REG 
        final_model = fit_models([final_model_func], stacked_train_X, y_train_total)[0]
        predict_dict = predict_models([final_model], stacked_train_X, stacked_val_X, stacked_test_X)

    elif use_blending:
        blended_train_X = stacking_pred(train_pred_prob).reshape(-1,1)
        blended_val_X = stacking_pred(val_pred_prob).reshape(-1,1)
        blended_test_X = stacking_pred(test_pred_prob).reshape(-1,1)

        final_model_func = ModelFunc.LOGISTIC_REG
        final_model = fit_models([final_model_func], blended_train_X, y_train_total)[0]
        predict_dict = predict_models([final_model], blended_train_X, blended_val_X, blended_test_X)

    else:
        final_model_func = params['model_funcs'][0]
        final_model = fit_models([final_model_func], train_pred_prob.reshape(-1,1), y_train_total)[0]
        predict_dict = predict_models([final_model], train_pred_prob.reshape(-1,1), val_pred_prob.reshape(-1,1),\
                                       test_pred_prob.reshape(-1,1))

    ensemble_train = [d['train'] for d in predict_dict][0]
    ensemble_val = [d['val'] for d in predict_dict][0]
    ensemble_test = [d['test'] for d in predict_dict][0]

    # ## Display metrics, ROC AUC for train, val and test samples
    train_roc_auc = roc_auc_score_metric(y_train_total, ensemble_train)
    val_roc_auc = roc_auc_score_metric(y_val_total, ensemble_val)
    test_roc_auc = roc_auc_score_metric(y_test_total, ensemble_test)
  
    print('=== Train sample metrics ===')
    print(f'ROC AUC: {train_roc_auc:.4f}')
    print(calculate_metrics_table(y_train_total, ensemble_train))

    print('=== Val sample metrics ===')
    print(f'ROC AUC: {val_roc_auc:.4f}')
    print(calculate_metrics_table(y_val_total, ensemble_val))

    print('=== Test sample metrics ===')
    print(f'ROC AUC: {test_roc_auc:.4f}')
    print(calculate_metrics_table(y_test_total, ensemble_test))
    print('===========================')  

def objective_CatBoostClassifier(trial, X_train, y_train, X_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
        'rsm': trial.suggest_uniform('rsm', 0.5, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 50,
        # 'cv': 5,
        # 'task_type': 'GPU', # error with subsample
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    # Штраф за переобучение (разница между тренировочной и валидационной метриками)
    overfitting_penalty = abs(train_auc - val_auc)
    # Целевая функция с учетом штрафа: при переобучении функция уменьшится
    score = val_auc - overfitting_penalty
    
    return score

def optuna_study_CatBoostClassifier(X_train, y_train, X_val, y_val, n_trials=100):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_CatBoostClassifier(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, n_jobs=-1)
    return study

def optuna_plot_optimization_history(study):
    #plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
    return optuna.visualization.plot_optimization_history(study)
def optuna_plot_parallel_coordinate(study):
    #plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
    return optuna.visualization.plot_parallel_coordinate(study)
def optuna_plot_slice(study):
    #plot_slice: shows the evolution of the search. You can see where in the hyperparameter space the trials were exploring.
    return optuna.visualization.plot_slice(study)
def optuna_plot_param_importances(study):
    #plot_param_importances: shows the relative importances of hyperparameters.
    return optuna.visualization.plot_param_importances(study)
def optuna_plot_edf(study):
    #plot_edf: plots the empirical distribution function of the objective.
    return optuna.visualization.plot_edf(study)

def get_model_params(model_func):
    match(model_func):
        case ModelFunc.LINEAR_REG:
            return {
            }
        case ModelFunc.LOGISTIC_REG:
            return  {
                'solver': 'saga', # default liblinear,saga
                'C': 10.0, # default 1.0
                'penalty': 'elasticnet', # l1,l2,elasticnet
                'l1_ratio': 0.5,
                'max_iter': 1000, # default 100
                'tol': 1e-8,
            }
        case ModelFunc.SVC_CLASS:
            return  {
                'kernel': 'linear',
                'C': 10.0,             #Regularization params
                'max_iter': 1000,
                'random_state': 42,
            }
        case ModelFunc.CATBOOST_CLASS | ModelFunc.CATBOOST_REG:
            return {
                'n_estimators': 1000,       # Общее количество деревьев (итераций). Меньшее значение снижает вероятность переобучения.
                'random_state': 42,         # Устанавливает начальное значение для генератора случайных чисел, что обеспечивает воспроизводимость результатов.
                'max_depth': 6,             # Глубина каждого дерева. Меньшая глубина снижает вероятность переобучения.
                'learning_rate': 0.1,       # Темп обучения. Более низкое значение помогает улучшить стабильность и уменьшить вероятность переобучения.
                'l2_leaf_reg': 3.0,         # Коэффициент L2-регуляризации на веса в листьях. Увеличивает штраф за большие веса и снижает переобучение.
                'bagging_temperature': 1.0, # Параметр, контролирующий интенсивность случайности в выборке для каждого дерева. Чем выше значение, тем больше разнообразие деревьев.
                'rsm': 0.8,                 # Доля признаков, используемых при обучении каждого дерева. Значение меньше 1 уменьшает переобучение.
                'subsample': 0.8,           # but error # Доля данных, используемых для каждого дерева. Чем меньше значение, тем сильнее регуляризация и выше разнообразие деревьев.
                'early_stopping_rounds': 50,
                # 'task_type': 'GPU', # Error with rsm
                'verbose': 0,
            }
        case ModelFunc.CATBOOST_CLASS_GRID_SEARCH | ModelFunc.CATBOOST_REG_GRID_SEARCH:
            return {
                'n_estimators': [200, 300, 400],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1.0, 3.0, 5.0, 7.0],
                'bagging_temperature': [0, 0.3, 0.6, 1.0],
                'rsm': [0.6, 0.8, 1.0],
                'subsample': [0.6, 0.8, 1.0],

                'scoring': 'roc_auc',
                'cv': 3,
                'n_jobs': -1,
                'early_stopping_rounds': 50,
                'random_state': 42,
                'task_type': 'GPU',
                'verbose': 0,
            }
        case ModelFunc.XGBOOST_CLASS | ModelFunc.XGBOOST_REG:
            return {
                'n_estimators': 100,      
                'random_state': 42,       
                'learning_rate': 0.1,     
                'subsample':0.8,               #0.5-1.0
                # 'reg_alpha': 1.0             #L1 regularization term on weights (xgb's alpha).
                'reg_lambda': 3.0,             #L2 regularization term on weights (xgb's lambda).
                'colsample_bytree': 5.0,       #Subsample ratio of columns when constructing each tree.
                # 'colsample_bylevel': 3.0,    #ubsample ratio of columns for each level.
                'early_stopping_rounds': 50,
                'device': 'cuda',
                'verbosity': 0,
            }
        case ModelFunc.RANDOM_FOREST_CLASS | ModelFunc.RANDOM_FOREST_REG:
            return {
                'n_estimators': 200,     # Количество деревьев в лесу. Большее количество деревьев может улучшить точность, но увеличивает время обучения.
                'max_depth': 5,          # Максимальная глубина каждого дерева. Ограничение глубины снижает вероятность переобучения.
                'min_samples_split': 10, # Минимальное число образцов для разделения узла. Большее значение предотвращает разделение узлов с малым числом выборок.
                'min_samples_leaf': 5,   # Минимальное количество выборок, которое должно находиться в каждом листе. Увеличение значения делает модель более устойчивой.
                # 'max_features': 'sqrt',         # Максимальное количество признаков, используемых при поиске лучшего разбиения. "sqrt" берёт корень из общего числа признаков.
                # 'max_leaf_nodes': 20,           # Максимальное число листьев в каждом дереве. Ограничивает количество конечных узлов, упрощая структуру дерева.
                # 'min_impurity_decrease': 0.01,  # Минимальное уменьшение нечистоты, требуемое для разделения. Предотвращает создание слишком мелких узлов.
                # 'bootstrap': True,              # Использовать бутстрэп (выборка с возвращением) для создания деревьев. Это повышает устойчивость модели.
                'random_state': 42,      # Устанавливает начальное значение для генератора случайных чисел, что обеспечивает воспроизводимость результатов.
            }
        case ModelFunc.DECISION_TREE_CLASS | ModelFunc.DECISION_TREE_REG:
            return  {
                'max_depth': 4,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_leaf_nodes': 15,  
                'random_state': 42,
            }
        case ModelFunc.DECISION_TREE_CLASS_GRID_SEARCH | ModelFunc.DECISION_TREE_REG_GRID_SEARCH:
            return  {
                'max_depth': [4, 6, 8,],
                'min_samples_split': [5, 7, 9],
                'min_samples_leaf': [3, 5, 7],
                'max_leaf_nodes': [5, 10, 15],  
                'random_state': 42,
                'scoring': 'roc_auc',       # Оценочная метрика для выбора наилучшей модели
                'cv': 3,                    # Количество фолдов для кросс-валидации
                'n_jobs': -1,               # Параллельное выполнение
            }
        case ModelFunc.KNN_CLASS | ModelFunc.KNN_REG:
            return  {
                # 'n_neighbors': 200, # default 5
                'algorithm': 'kd_tree', #default auto
                'p': 1, #default 2
                # 'metric': 'minkowski', # minkowski
                'n_jobs': 1,
            }
        case ModelFunc.LSTM_CLASS:
            return  {
                'n_steps': 30,
                'n_features': 8,
                'batch_size': 15,
                'epochs': 30,
                'units': 150,
                'activation': 'linear', #relu
                'validation_split': 0.1,
                'optimizer': 'adam',
                'loss': 'mse',
                'return_sequences': True,
                'shuffle': True,
                'verbose': 0,
            }
        case _:
            print(f'Unknown {model_func}')

class ModelFunc(Enum):
    LOGISTIC_REG = logistic_regression_model
    LINEAR_REG = linear_regression_model
    RANDOM_FOREST_CLASS = random_forest_classifier_model
    RANDOM_FOREST_REG = random_forest_regressor_model
    DECISION_TREE_CLASS = decision_tree_classifier_model
    XGBOOST_CLASS = xgboost_classifier_model
    XGBOOST_REG = xgboost_regressor_model
    CATBOOST_CLASS = catboot_classifier_model
    CATBOOST_REG = catboot_regressor_model
    KNN_CLASS = knn_classifier_model
    KNN_REG = knn_regressor_model
    SVC_CLASS = svc_classifier_model
    LSTM_CLASS = lstm_model
 
    DECISION_TREE_CLASS_GRID_SEARCH = decision_tree_classifier_model_grid_search
    DECISION_TREE_REG_GRID_SEARCH = decision_tree_regressor_model_grid_search
    CATBOOST_CLASS_GRID_SEARCH = catboot_classifier_model_grid_search
    CATBOOST_REG_GRID_SEARCH = catboot_regressor_model_grid_search


if __name__ == "__main__":
    pass
    # params = get_model_params(model_func)
    # print(f'params: {params}')
    # X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)
    # model = model_fit(model_func, X, y, params)
    # print(f'model: {model}')