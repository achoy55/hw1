import os
import data_processing as dp
import pandas as pd

from datetime import datetime
from data_loader import load_data_period
from tqdm import tqdm


def _test(tickers):
    dates = list()
    dates.append({
        'from': datetime(2019, 1, 1),
        'to': datetime(2021, 1, 1),
    })
    dates.append({
        'from': datetime(2020, 1, 1),
        'to': datetime(2024, 1, 1),
    })

    time_interval='1d'

    file_compress = True
    for period in tqdm(dates):
        print(f'Start download data, period: {period}')
        crypto_dir = load_data_period(tickers, period['from'], period['to'], time_interval) # default crypto_dir

        for ticker in tickers:
            print(f'Start merge, ticker: {ticker}, {crypto_dir}')

            new_df = dp.get_data(crypto_dir, ticker, compress=False) # get new raw data
            print(f'New data: {len(new_df)}')

            merged_df = dp.merge_and_store_new_data(new_df, ticker, compress=file_compress) # merge with old and check duplication
            print(f'Merged data: {len(new_df)}')

            dp.store_to_file(merged_df, ticker, compress=file_compress) # save merged
            yield ( ticker, merged_df )

def _test_all(tickers):
    time_interval='1d'

    file_compress = True
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 1, 1)
    print(f'Start download data, period: {start_date} - {end_date}')
    crypto_dir = load_data_period(tickers, start_date, end_date, time_interval)

    for ticker in tqdm(tickers):
        print(f'Start merge, ticker: {ticker}, {crypto_dir}')

        new_df = dp.get_data(crypto_dir, ticker, compress=False) # get new raw data
        print(f'New data: {len(new_df)}')

        merged_df = dp.merge_and_store_new_data(new_df, ticker, compress=file_compress) # merge with old and check duplication
        print(f'Merged data: {len(new_df)}')
    
        dp.store_to_file(merged_df, ticker, compress=file_compress) # save merged
        yield ( ticker, merged_df )


if __name__ == "__main__":
    os.system('rm -rf crypto_dir/*')
    os.system('rm -rf _data_store/*')

    tickers = ['BTC-USD', 'ETH-USD'] #, 'SOL-USD', 'XRP-USD'

    ## Load all data 
    result = list()
    for ticker, df in tqdm(_test_all(tickers)):
        print(df.isnull().sum())
        print(f'=== {ticker}, duplicated: {df.index.duplicated().sum()}')

    ## Get all stored data
    for ticker in tqdm(tickers):
        saved_data = dp.get_data('_data_store', ticker, compress=True)
        result.append({
            'ticker': ticker,
            'len_df': len(df),
        })

    os.system('rm -rf crypto_dir/*')
    os.system('rm -rf _data_store/*')

    # ===== by period and merge ======
    result2 = list()
    for ticker, df in _test(tickers):
        print(df.isnull().sum())
        print(f'=== {ticker}, duplicated: {df.index.duplicated().sum()}')

    for ticker in tqdm(tickers):
        saved_data = dp.get_data('_data_store', ticker, compress=True)
        result2.append({
            'ticker': ticker,
            'len_df': len(df),
        })

    print('=== FINAL ===')
    res_df1 = pd.DataFrame.from_dict(result)
    res_df2 = pd.DataFrame.from_dict(result2)
    print(f"is equals: {res_df1['len_df'].equals(res_df2['len_df'])}")
    

