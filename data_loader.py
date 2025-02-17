import yfinance as yf
from datetime import timedelta, datetime
import os

print("yfinance version: {}".format(yf.__version__))


def _create_data_folder(dir):
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

def _load_data_at_start_date(tickers, period, time_interval, dir):
    dir = _create_data_folder(dir)

    delta = timedelta(days=period)
    today = datetime.now()
    print(f"date: {today+delta}")

    data = yf.download(tickers, today+delta, interval=time_interval, group_by='ticker')
    data.index.names = ['Date']
    
    return data

def _load_data_by_date_range(tickers, start_date, end_date, time_interval, dir):
    dir = _create_data_folder(dir)

    data = yf.download(tickers, start_date, end_date, interval=time_interval, group_by='ticker')
    data.index.names = ['Date']
  
    return data

def load_data_at_start_date(tickers, period, time_interval, dir):
    dir = _create_data_folder(dir)

    print(f'Start load data, tickers {tickers}, interval: {time_interval}, start date: {period}')

    data = _load_data_at_start_date(tickers, period, time_interval, dir)
    data.index.names = ['Date']
    
    _save_data(tickers, data, dir)
    data.info()

    print('Download data completed')
    return data

def load_data_by_period(tickers, period, time_interval, dir='crypto_data'):
    dir = _create_data_folder(dir)

    print(f'Start load data, tickers {tickers}, interval: {time_interval}, period: {period}')

    data = yf.download(tickers, period=period, interval=time_interval, group_by='ticker')
    data.index.names = ['Date']
    
    _save_data(tickers, data, dir)
    data.info()

    print('Download data completed')
    return dir

def load_data_by_date_range(tickers, start_date, end_date, time_interval, dir='crypto_data'):
    dir = _create_data_folder(dir)
    
    print(f'Start load data for ticker {tickers}, period [{start_date} - {end_date}], interval: {time_interval}')

    data = _load_data_by_date_range(tickers, start_date, end_date, time_interval, dir)
    data.index.names = ['Date']
  
    _save_data(tickers, data, dir)
    data.info()

    print('Download data completed')
    return dir

def load_forex_data_at_start_date(tickers, period, time_interval, dir='forex_data'):
    dir = _create_data_folder(dir)

    print(f'Start load forex data, tickers {tickers}, interval: {time_interval}, from: {period}')

    data = _load_data_at_start_date(tickers, period, time_interval)
    data.index.names = ['Date']
    
    _save_forex_data(tickers, data, dir)
    data.info()

    print('Download forex data completed')
    return dir

def load_forex_data_period(tickers, start_date, end_date, time_interval, dir='forex_data'):
    dir = _create_data_folder(dir)
    
    print(f'Start load forex data for ticker {tickers}, period [{start_date} - {end_date}], interval: {time_interval}')

    data = _load_data_by_date_range(tickers, start_date, end_date, time_interval)
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
