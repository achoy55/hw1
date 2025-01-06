import yfinance as yf
import pandas as pd
from datetime import timedelta, datetime
import os

print("yfinance version: {}".format(yf.__version__))


def _create_data_folder(dir = 'crypto_data'):
    #Should be ${workspaceFolder} in vscode://settings/jupyter.notebookFileRoot 
    if not os.path.exists(dir) :
        os.makedirs(dir)
        print(f'Directory [{dir}] created')
    return dir

def _save_data(tickers, data, save_location):
    for i in tickers:
        try:
            TEMP = data[i].copy(deep=True)
            TEMP = TEMP.dropna()
            TEMP.to_csv(save_location+"/"+i+".csv")
        except:
            print(f'Unable load data for ticker {i}')

def _save_forex_data(tickers, data, save_location):
    for i in tickers:
        symbol = i[:len(i)-2]
        try:
            TEMP = data[i].copy(deep=True)
            TEMP = TEMP.dropna()
            TEMP.to_csv(save_location+"/"+symbol+".csv")
        except:
            print(f'Unable load {i} data')

def load_crypto_data(tickers, period, time_interval):
    dir = _create_data_folder()

    delta = timedelta(days=period) # 
    today = datetime.now()

    print(f'Start load crypto data, tickers {tickers}, interval: {time_interval}, from: {today+delta}')

    data = yf.download(tickers, today+delta, interval=time_interval, group_by='ticker')
    data.index.names = ['Date']
    
    _save_data(tickers, data, dir)
    data.info()

    return dir

def load_crypto_data2(tickers, start_date, end_date, time_interval):
    dir = _create_data_folder()
    print(f'Start load crypto data for ticker {tickers}, period [{start_date} - {end_date}], interval: {time_interval}')

    data = yf.download(tickers, start_date, end_date, interval=time_interval, group_by='ticker')
    data.index.names = ['Date']
  
    _save_data(tickers, data, dir)
    data.info()

    print('Download crypto data completed')
    return dir

def load_forex_data(tickers, period, time_interval):
    dir = _create_data_folder('forex_data')

    delta = timedelta(days=period)
    today = datetime.now()

    print(f'Start load forex data, tickers {tickers}, interval: {time_interval}, from: {today+delta}')

    data = yf.download(tickers, today+delta, interval=time_interval, group_by='ticker')
    data.index.names = ['Date']
    
    _save_forex_data(tickers, data, dir)
    data.info()

    print('Download forex data completed')
    return dir

def load_forex_data2(tickers, start_date, end_date, time_interval):
    dir = _create_data_folder('forex_data')
    print(f'Start load forex data for ticker {tickers}, period [{start_date} - {end_date}], interval: {time_interval}')

    data = yf.download(tickers, start_date, end_date, interval=time_interval, group_by='ticker')
    data.index.names = ['Date']
  
    _save_forex_data(tickers, data, dir)
    data.info()

    print('Download forex data completed')
    return dir

if __name__ == "__main__":
    pass
    # period=-(datetime(2024,12,27) - datetime(2019, 1, 1)).days
    # time_interval='1d'
    # tickers = ['BTC-USD', 'ETH-USD'] #, 'SOL-USD', 'XRP-USD'
    # crypto_dir = load_data(tickers, datetime(2019, 1, 1), datetime(2024, 12, 28), time_interval)
    # print(crypto_dir)
