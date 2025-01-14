import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
from pykalman import KalmanFilter
from dataprep.eda import create_report
from dataprep.clean import clean_df

# служебные функции
from adtk.data import validate_series
from adtk.visualization import plot
# Статистические методы детектирования точечных аномалий
from adtk.detector import ThresholdAD
from adtk.detector import QuantileAD
from adtk.detector import InterQuartileRangeAD
from adtk.detector import GeneralizedESDTestAD
# Статистические методы детектирования групповых аномалий
from adtk.detector import PersistAD
from adtk.detector import LevelShiftAD
from adtk.detector import VolatilityShiftAD
# методы на основе декомпозиции временного ряда и авторегрессии
from adtk.detector import SeasonalAD
from adtk.detector import AutoregressionAD
# Методы на основе кластеризации - неконтролируемое обучение
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
# Методы на основе плотности
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor
# Методы на основе регрессии - контролируемое обучение
from adtk.detector import RegressionAD
from sklearn.linear_model import LinearRegression
# Методы на основе понижения размерности
from adtk.detector import PcaAD
# кастомизация
from adtk.detector import CustomizedDetectorHD
from adtk.transformer import ClassicSeasonalDecomposition
from adtk.pipe import Pipeline


def detect_ThresholdAD(df, high, low):
    s = validate_series(df)
    threshold_ad = ThresholdAD(high=high, low=low)
    return threshold_ad.detect(s)

def detect_QuantileAD(df, high, low):
    s = validate_series(df)
    quantile_ad = QuantileAD(high=high, low=low)
    return quantile_ad.fit_detect(s)

def detect_InterQuartileRangeAD(df, c):
    s = validate_series(df)
    iqr_ad = InterQuartileRangeAD(c=c)
    return iqr_ad.fit_detect(s)

def detect_SeasonalAD(df, c=3.0, side="both"):
    s = validate_series(df)
    seasonal_ad = SeasonalAD(c=c, side=side)
    return seasonal_ad.fit_detect(s)

def detect_AutoregressionAD(df, n_steps=7*2, step_size=24, c=3.0):
    s = validate_series(df)
    autoregression_ad = AutoregressionAD(n_steps=n_steps, step_size=step_size, c=c)
    return autoregression_ad.fit_detect(s)

def detect_MinClusterDetector(df, n_clusters=3):
    s = validate_series(df)
    min_cluster_detector = MinClusterDetector(KMeans(n_clusters=n_clusters))
    return min_cluster_detector.fit_detect(s)

def detect_OutlierDetector(df, contamination=0.05):
    s = validate_series(df)
    outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=contamination))
    return outlier_detector.fit_detect(s)

def detect_RegressionAD(df, target='Close', c=3.0):
    s = validate_series(df)
    regression_ad = RegressionAD(regressor=LinearRegression(), target=target, c=c)
    return regression_ad.fit_detect(s)

def detect_anomalies_z(df, score, column_name):
    df_copy = df.copy()
    df_copy['Z-score'] = stats.zscore(df_copy[column_name])
    anomalies = df_copy[abs(df_copy['Z-score']) > score]
    return anomalies

def merge_anomalies_z(df, column_name, score):
    detect_anomalies = detect_anomalies_z(df, column_name, score)
    anomalies = pd.concat(anomalies, detect_anomalies)
    return anomalies

def normalize_with_zcore(df, window, score):
    clean_function = lambda x: x[np.abs(stats.zscore(x)) < score]
    return df.rolling(window=window).apply(clean_function)

def risk_rating_z_between_column(df, close_column, volume_column):
    df_copy = df.copy()
    anomalies_df_close = stats.zscore(df_copy[close_column])
    anomalies_df_volume = stats.zscore(df_copy[volume_column])
    # оценка риска
    close_risk = anomalies_df_close['Z-score'].apply(lambda x: abs(x).mean())
    volume_risk = anomalies_df_volume['Z-score'].apply(lambda x: abs(x).mean())

    total_risk = close_risk + volume_risk

    # нормализация
    return (total_risk - total_risk.min()) / (total_risk.max() - total_risk.min())

def plot_anomalies_by_column(data, anomalies, y_column):
    anomaly_df = anomalies[anomalies==True]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[y_column], mode='lines', name='Value'))
    fig.add_trace(go.Scatter(x=anomaly_df.index, y=data[y_column], mode='markers', name='Anomaly',
                             marker=dict(color='red')))
    fig.show()

def plot_anomalies(data, anomalies):
    anomaly_df = anomalies[anomalies==True]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, name='value', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=anomaly_df.index, y=data, name='anomaly', mode='markers'))
    fig.update_layout(title='Anomalies', xaxis_title='Date', yaxis_title='Value')
    fig.show()

def plot_anomalies_custom(df, y_column, config):
    figure = go.Figure()
    # plot baseline     
    figure.add_trace(go.Scatter(name=y_column, x = df.index, y=df[y_column], marker=dict(color='green')))
         
    # plot anomaly points     
    anomaly_df = df
    anomaly_df = anomaly_df[anomaly_df[config['anomaly_column']]==True]
            
    figure.add_trace(go.Scatter(name=config['legend_name'], x = anomaly_df.index, y=anomaly_df[y_column], 
        mode='markers',
        marker=dict(color=config['color'],size=10)))

    figure.update_layout(title= 'Anomalies', xaxis_title='date', yaxis_title='value', legend_title="Anomaly Type",)
    
    figure.show()

def kalman_filter(df, column_name='Close'):
    data = df.copy()
    kf = KalmanFilter(transition_matrices = [1], observation_matrices = [1], initial_state_mean = 0,
                  initial_state_covariance = 1, observation_covariance=1, transition_covariance=.01)
    state_means, _ = kf.filter(data[column_name])
    return state_means

def clean_data_dataprep(df, is_show_report=False):
    inferred_dtypes, cleaned_df = clean_df(df)
    print(inferred_dtypes, cleaned_df)

    if is_show_report:
        create_report(cleaned_df)
    return cleaned_df


if __name__ == "__main__":
    pass

    # ax = df_smoothed.plot(title='Kalman Filter', figsize=(14,6), lw=1, rot=0)
    # ax.set_xlabel('')
    # ax.set_ylabel('BTC-USD')
    # plt.tight_layout()
    # sns.despine()

    # anomalies = detect_QuantileAD(df['Volume'], 0.99, 0.01)
    # anomalies = detect_InterQuartileRangeAD(df['Volume'], 1.5)
    # anomalies = detect_ThresholdAD(df['Volume'], 30, 15)
    # anomalies = detect_InterQuartileRangeAD(df['Volume'], 1.5)
    # anomalies = detect_AutoregressionAD(close_df, 7*2, 6, 3.0)
    # anomalies = detect_SeasonalAD(df['Volume'], 3.0, 'both')

    # # plot(df['Volume'], anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2);

    # # anomalies.value_counts()
    # # anomalies.window = 30

    # plot_anomalies_by_column(df, anomalies, 'Volume')
    # plot_anomalies(df['Volume'], anomalies)



    # # anomalies = detect_MinClusterDetector(df, 3) # must be pandas Dataframe
    # # anomalies = detect_OutlierDetector(df, 0.05) # ERROR
    # # anomalies = detect_RegressionAD(df, 'Volume', 3.0) # ERROR
    # # plot(df, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all');

    # # print(anomalies.value_counts())
    # # plot(close_df, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, ts_color='g', anomaly_color='red', anomaly_tag="marker");

