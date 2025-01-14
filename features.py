import math
import numpy as np
from scipy import stats
from scipy.stats import linregress
import pandas as pd
import talib as ta
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    def __init__(self, params):
        self.params = params
 
    def create_trend_features(self, df, features, lag_periods):
        """
        Добавляет классические финансовые признаки: отношение к предыдущим периодам, 
          логарифмические изменения и индикаторы трендов.
        
        df: DataFrame с исходными данными
        features: список признаков, для которых необходимо добавить индикаторы
        lag_periods: сколько периодов назад учитывать для расчетов
        
        Возвращает:
        - обновленный DataFrame с новыми фичами
        - список новых колонок, которые можно использовать как признаки
        """
        data = df.copy()
        new_columns = list()  # Список для хранения новых колонок
        
        for feature in features:
            # Отношение текущего значения к предыдущему (лаг = 1)
            data[f'{feature}_ratio_1'] = data[feature] / data[feature].shift(1)
            new_columns.append(f'{feature}_ratio_1')
            
            # Логарифмическое изменение (логарифм отношения текущего значения к предыдущему)
            data[f'{feature}_log_diff_1'] = np.log(data[feature] / data[feature].shift(1))
            new_columns.append(f'{feature}_log_diff_1')
            
            # Momentum (разница между текущим значением и значением N периодов назад)
            data[f'{feature}_momentum_{lag_periods}'] = data[feature] - data[feature].shift(lag_periods)
            new_columns.append(f'{feature}_momentum_{lag_periods}')
            
            # Rate of Change (ROC): процентное изменение за N периодов
            data[f'{feature}_roc_{lag_periods}'] = (data[feature] - data[feature].shift(lag_periods)) / \
                                                    data[feature].shift(lag_periods) * 100
            new_columns.append(f'{feature}_roc_{lag_periods}')
            
            # Exponential Moving Average (EMA) с периодом N
            data[f'{feature}_ema_{lag_periods}'] = data[feature].ewm(span=lag_periods, adjust=False).mean()
            new_columns.append(f'{feature}_ema_{lag_periods}')

        # data['chg'] = data['Close'].pct_change().fillna(0).multiply(100).round(2)
        # data['vol_chg'] = data['Volume'].pct_change().fillna(0).multiply(100).round(2)

        # print(data.tail(10))

        data.dropna(inplace=True)
        return data, new_columns
    
    def create_rolling_features(self, df, features, window_sizes):
        """
        Statistics: Добавляет скользящие характеристики для указанных признаков и окон.
        
        df: DataFrame с исходными данными
        features: список признаков, для которых необходимо добавить скользящие характеристики
        window_sizes: список размеров окон для расчета характеристик (например, [5, 14, 30])
        
        Возвращает:
        - обновленный DataFrame с новыми фичами
        - список новых колонок, которые можно использовать как признаки
        """
        data = df.copy()  
        new_columns = []  # Список для хранения новых колонок
        
        # Для каждого признака и для каждого окна
        for feature in features:
            for window_size in window_sizes:
                # Скользящее среднее
                new_col_name = f'{feature}_mean_{window_size}'
                data[new_col_name] = data[feature].rolling(window=window_size).mean()
                new_columns.append(new_col_name)
                
                # Скользящая медиана
                new_col_name = f'{feature}_median_{window_size}'
                data[new_col_name] = data[feature].rolling(window=window_size).median()
                new_columns.append(new_col_name)
                
                # Скользящий минимум
                new_col_name_min = f'{feature}_min_{window_size}'
                data[new_col_name_min] = data[feature].rolling(window=window_size).min()
                new_columns.append(new_col_name_min)
                
                # Скользящий максимум
                new_col_name_max = f'{feature}_max_{window_size}'
                data[new_col_name_max] = data[feature].rolling(window=window_size).max()
                new_columns.append(new_col_name_max)
                
                # Скользящее стандартное отклонение
                new_col_name = f'{feature}_std_{window_size}'
                data[new_col_name] = data[feature].rolling(window=window_size).std()
                new_columns.append(new_col_name)
                
                # Скользящий размах (макс - мин)
                new_col_name = f'{feature}_range_{window_size}'
                data[new_col_name] = data[new_col_name_max] - data[new_col_name_min]
                new_columns.append(new_col_name)
                
                # Скользящее абсолютное отклонение от медианы (mad)
                new_col_name = f'{feature}_mad_{window_size}'
                data[new_col_name] = data[feature].rolling(window=window_size).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
                new_columns.append(new_col_name)
        
        data.dropna(inplace=True)
        return data, new_columns

    def create_lag_features(self, df, features, lag_periods):
        """
        Добавляет лаги для указанных признаков на указанное количество периодов назад.
        
        df: DataFrame с исходными данными
        features: список признаков, для которых необходимо добавить лаги
        lag_periods: сколько лагов назад необходимо создать
        Возвращает: 
        - обновленный DataFrame с лагами
        - список новых колонок, которые можно использовать как признаки
        """
        data = df.copy()
        new_columns = []  # Список для хранения новых колонок
        
        # Для каждого признака создаем лаги
        for feature in features:
            for lag in range(1, lag_periods + 1):
                new_col_name = f'{feature}_lag_{lag}'
                data[new_col_name] = data[feature].shift(lag)
                new_columns.append(new_col_name)
        
        data.dropna(inplace=True)
        return data, new_columns

    def create_lag_and_target_forward_returns(self, df):
        data = df.copy()

        lags = self.params['lags']
        q = self.params['quantile']

        # For example [1, 5, 10, 21, 42, 63]
        for lag in lags:
            data[f'return_{lag}d'] = data.Close.pct_change(lag) \
                            .pipe(lambda x: x.clip(lower=x.quantile(q),upper=x.quantile(1 - q))) \
                            .add(1) \
                            .pow(1 / lag) \
                            .sub(1)

        # Lag, shift the daily, (bi-)weekly, and monthly returns to use them as features
        for t in [1, 2, 3, 4, 5]:
            for lag in [1, 5, 10, 21]:
                data[f'return_{lag}d_lag{t}'] = data[f'return_{lag}d'].shift(t * lag)

        """
        More specifically, we shift returns for time horizon t back by t days to use them as forward returns.
        For instance, we shift the 5-day return from t0 to t5 back by 5 days so that this value
         becomes the model target for t0.
        We can generate daily, (bi-)weekly, and monthly forward returns
        """
        for t in [1, 5, 10, 21]:
            data[f'target_{t}d'] = data[f'return_{t}d'].shift(-t)
        
        return data
        
        
        

    def get_attributes(df):
        attributes = list(df.columns)
        if 'Date' in attributes: attributes.remove('Date')
        return attributes
    
    def get_slope(array):
        y = np.array(array)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return slope

    def enrich_with_indicators(self, df):
        data = df.copy()
        rsi_period = 14
        macd_fast_period = 12
        macd_slow_period = 26
        macd_signal_period = 9
        emaf_period = 20
        emam_period = 100
        emas_period = 150
        atr_period = 20
        avg_period = 14
        adx_period = 14
        dmi_period = 14
        bbands_period = 21

        if self.params.get('rsi') != None:
            rsi_period = self.params['rsi']
        if self.params.get('macd') != None:
            macd_fast_period = self.params['macd'][0]
            macd_slow_period = self.params['macd'][1]
            macd_signal_period = self.params['macd'][2]
        if self.params.get('emaf') != None:
            emaf_period = self.params['emaf']
        if self.params.get('emam') != None:
            emam_period = self.params['emam']
        if self.params.get('emas') != None:
            emas_period = self.params['emas']
        if self.params.get('atr') != None:
            emas_period = self.params['atr']
        if self.params.get('adx') != None:
            adx_period = self.params['adx']
        if self.params.get('dmi') != None:
            dmi_period = self.params['dmi']
        if self.params.get('bbands') != None:
            bbands_period = self.params['bbands']

        h = data['High']
        l = data['Low']
        c = data['Close']

        # Relative Strength Index, analyze overbought or oversold conditions.
        data['rsi'] = ta.RSI(c, timeperiod=rsi_period) / ta.RSI(c, timeperiod=rsi_period).mean()
        data['emaf']= ta.EMA(c, timeperiod=emaf_period) / ta.EMA(c, timeperiod=emaf_period).mean()
        data['emam']= ta.EMA(c, timeperiod=emam_period) / ta.EMA(c, timeperiod=emam_period).mean()
        data['emas']= ta.EMA(c, timeperiod=emas_period) / ta.EMA(c, timeperiod=emas_period).mean()
        data['macd']= ta.MACD(c, fastperiod=macd_fast_period, slowperiod=macd_slow_period, signalperiod=macd_signal_period)[0] / \
                    ta.MACD(c, fastperiod=macd_fast_period, slowperiod=macd_slow_period, signalperiod=macd_signal_period)[0].mean()
        # Average Directional Index, Trend Strength, price trend
        data['adx'] = ta.ADX(h, l, c, timeperiod=adx_period) / ta.ADX(h, l, c, timeperiod=adx_period).mean()
        # Average Directional Movement Index Rating 
        data['adxr'] = ta.ADXR(h, l, c, timeperiod=adx_period) / ta.ADXR(h, l, c, timeperiod=adx_period).mean()
        # The Average True Range, market volatility, risk management
        data['atr'] = ta.ATR(h, l, c, timeperiod=atr_period) / ta.ATR(h, l, c, timeperiod=atr_period).mean()
        
        # Plus/Minus Directional Indicator 
        # data['+dmi'] = ta.PLUS_DI(h, l, c ,timeperiod=dmi_period)
        # data['-dmi'] = ta.MINUS_DI(h, l, c,timeperiod=dmi_period)

        # Bollinger Bands
        # data['bbands'] = ta.BBANDS(c, timeperiod=bbands_period, nbdevup=2, nbdevdn=2, matype=0)[0] / \
        #                 ta.BBANDS(c, timeperiod=bbands_period, nbdevup=2, nbdevdn=2, matype=0)[0]

        # data['average'] = ta.MIDPRICE(h, l, timeperiod=avg_period) / ta.MIDPRICE(h, l, timeperiod=avg_period).mean()
        # data['ma40'] = ta.SMA(c, timeperiod=40)
        # data['ma80'] = ta.SMA(c, timeperiod=80)
        # data['ma160'] = ta.SMA(c, timeperiod=160)
        # window = 6
        # data['slopeMA40'] = df['MA40'].rolling(window=window).apply(get_slope, raw=True)
        # data['slopeMA80'] = df['MA80'].rolling(window=window).apply(get_slope, raw=True)
        # data['slopeMA160'] = df['MA160'].rolling(window=window).apply(get_slope, raw=True)
        # data['AverageSlope'] = df['Average'].rolling(window=window).apply(get_slope, raw=True)
        # data['RSISlope'] = df['RSI'].rolling(window=window).apply(get_slope, raw=True)

        data.dropna(inplace=True)
        return data
    
    def add_target(self, df, period):
        data = df.copy()
        data['TargetClass'] = data['Close'].shift(-period)
        data['Target'] = (data['TargetClass'] > data['Close']).astype(int)
        data.dropna(subset=['TargetClass'], inplace=True)
        return data

    def add_target2(self, df):
        data = df.copy()
        data['TargetClass'] = data['Close']-data.Open
        data['TargetClass'] = data['TargetClass'].shift(-1)
        data['Target'] = [1 if data.iloc[i].TargetClass>0 else 0 for i in range(len(data))]
        data['TargetNextClose'] = data['Close'].shift(-1)
        return data

    def clear_invalid_targets(self, df):
        data = df.copy()
        return data.dropna(subset=['TargetClass', 'Target'])
        
    def validate_outliers(self, df, column_name, min_outliers=.25, max_outliers=.75):
        data = df.copy()
        outliers = self._find_outliers_IQR(data[column_name], min_outliers, max_outliers)
        print('Outliers detected:', len(outliers))
        if len(outliers) > 0:
            return self._drop_outliers_IQR(data, column_name, min_outliers, max_outliers)
        return data;
    
    def _find_outliers_IQR(self, data, min_outliers, max_outliers):
        lower, upper = self._calc_IQR(data, min_outliers, max_outliers)
        return data[(data<lower) | (data>upper)]

    def _drop_outliers_IQR(self, data, column_name, min_outliers, max_outliers):
        """
        Drop outlier data
        """
        lower, upper = self._calc_IQR(data[column_name], min_outliers, max_outliers)
        upper_array = np.where(data[column_name] >= upper)[0]
        lower_array = np.where(data[column_name] <= lower)[0]
        print('Dropping', len(upper_array), 'upper outliers')
        print('Dropping', len(lower_array), 'lower outliers')

        data.drop(index=upper_array, inplace=True)
        data.drop(index=lower_array, inplace=True)
        return data
    
    def _calc_IQR(self, data, min_outliers, max_outliers):
        """
        Calculate quantile outliers
        """
        q1=data.quantile(min_outliers)
        q3=data.quantile(max_outliers)
        IQR=q3-q1
        lower = q1 - 1.5*IQR
        upper = q3 + 1.5*IQR
        return lower, upper


if __name__ == "__main__":
    pass