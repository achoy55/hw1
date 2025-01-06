import pandas as pd
import numpy as np
from enum import Enum

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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


def validate_duplicate_and_merge(df1, df2):
    """
        Validate and merge two dataframes, index as 'Date'
    """
    if not df1.empty and type(df1.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df1.set_index('Date', inplace=True)
    if type(df2.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df2.set_index('Date', inplace=True)
    merged_df = pd.concat([df1, df2])
    return merged_df[~merged_df.index.duplicated(keep='first')]

def detect_anomalies_adtk(df, column_name):
    anomalies = detect_SeasonalAD(df[column_name], 3.0, 'both')
    print('Anomalies detected:', anomalies.value_counts())
    return anomalies

def linear_regression_model(X_train, y_train, params):
    # true_weights = [2, 0.00001]
    # y = true_weights[0] * X_train + true_weights[1] * y_train + np.random.normal(0, 1, len(y_train))
    
    model_params = params.copy()
    model = LinearRegression().set_params(**model_params)
    model.fit(X_train, y_train)
    return model

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
        model = CatBoostClassifier(**model_params)
    if ModelFunc.CATBOOST_REG is model_func:
        model = CatBoostRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=eval_set)
    return model

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
            if model_func is ModelFunc.CATBOOST_CLASS:
                params = dict(params, early_stopping_rounds=50)
                model = model_fit_with_eval(model_func, X_train, y_train, eval_set=(X_val, y_val), params=params)
            else:
                model = model_fit(model_func, X_train, y_train, params)
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

def split_data_by_date(data_with_features):
    test_start_date = pd.to_datetime(data_with_features.index.max()) - pd.DateOffset(months=1)
    val_start_date = pd.to_datetime(data_with_features.index.max()) - pd.DateOffset(months=2)

    train_data = data_with_features[pd.to_datetime(data_with_features.index) < val_start_date] 
    val_data = data_with_features[(pd.to_datetime(data_with_features.index) >= val_start_date) & \
                                  (pd.to_datetime(data_with_features.index) < test_start_date)]
    test_data = data_with_features[pd.to_datetime(data_with_features.index) >= test_start_date]
    # print(data_with_features.index[-1], train_data.index[-1], val_data.index[-1], test_data.index[-1])    

    return train_data, val_data, test_data

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

def get_model_params(model_func):
    match(model_func):
        case ModelFunc.LINEAR_REG:
            return {
            }
        case ModelFunc.LOGISTIC_REG:
            return  {
                'solver': 'liblinear', # default liblinear
                'C': 0.1, # default 1.0
                'max_iter': 1000, # default 100
            }
        case ModelFunc.SVC_CLASS:
            return  {
                'kernel': 'linear',
                'C': 0.1,
                'max_iter': 1000,
                'random_state': 42,
            }
        case ModelFunc.CATBOOST_CLASS | ModelFunc.CATBOOST_REG:
            return {
                'n_estimators':1000,
                'random_state': 42,
                'max_depth': 6,
                'learning_rate': 0.01,
                'task_type': 'GPU',
                'verbose': 0,
            }
        case ModelFunc.XGBOOST_CLASS | ModelFunc.XGBOOST_REG:
            return {
                'n_estimators': 100,
                'random_state': 42,
                'learning_rate': 0.1,
                'device': 'cuda',
            }
        case ModelFunc.RANDOM_FOREST_CLASS | ModelFunc.RANDOM_FOREST_REG:
            return {
                'n_estimators': 200,
                'random_state': 42,
                # 'max_depth': 4,
            }
        case ModelFunc.DECISION_TREE_CLASS | ModelFunc.DECISION_TREE_REG:
            return  {
                'max_depth': 4,
                'random_state': 42,
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
    DECISION_TREE_REG = decision_tree_regressor_model
    XGBOOST_CLASS = xgboost_classifier_model
    XGBOOST_REG = xgboost_regressor_model
    CATBOOST_CLASS = catboot_classifier_model
    CATBOOST_REG = catboot_regressor_model
    KNN_CLASS = knn_classifier_model
    KNN_REG = knn_regressor_model
    SVC_CLASS = svc_classifier_model
    LSTM_CLASS = lstm_model


if __name__ == "__main__":
    pass
    # params = get_model_params(model_func)
    # print(f'params: {params}')
    # X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)
    # model = model_fit(model_func, X, y, params)
    # print(f'model: {model}')