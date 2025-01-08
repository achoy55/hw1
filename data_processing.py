import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go

def _loading_data(path: str) -> pd.DataFrame:
    renaming_dict = {
        'Adj Close': 'Close'
    }

    df = pd.read_csv(path, parse_dates=True, keep_date_col=True, index_col=0)
    if 'Adj Close' in df.columns:
        df.drop(columns=['Close'], inplace=True)
        df.rename(columns=renaming_dict, inplace=True)

    for col in df.columns[1:]:
        try:
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].astype('double')
        except AttributeError:
            continue

    return df

def get_data(dir: str, filename: str, compress=False):
    ext = '.csv'
    if compress:
        ext = '.zip'
            
    try:
        f = os.path.join(dir, filename+ext)
        if os.path.isfile(f):
            return _loading_data(f)
    except Exception as e:
        print(f"Error loading file {filename+ext}, dir: {dir}: {e}")
        raise
    else:
        return pd.DataFrame()

def _create_store_folder(store_dir = '_data_store'):
    os.makedirs(store_dir, exist_ok=True) 

def store_to_file(data, filename, store_dir = '_data_store', compress=False):
    _create_store_folder()

    file_path = store_dir+'/'+filename
    compression_opts = None
    if compress:
        compression_opts = dict(method='zip', archive_name=filename+'.csv')
        file_path += '.zip'
    else:
        file_path += '.csv'

    data.reset_index(inplace=True)
    data.to_csv(file_path, compression=compression_opts, index=False)

def validate_duplicate_and_merge(df1, df2):
    """
    Validate and merge two dataframes, index 'Date'
    """
    if not df1.empty and type(df1.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df1.set_index('Date', inplace=True)
    if type(df2.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df2.set_index('Date', inplace=True)
    merged_df = pd.concat([df1, df2])
    return merged_df[~merged_df.index.duplicated(keep='first')]
    
def merge_and_store_new_data(new_df, key, store_dir='_data_store', compress=False):
    saved_data = get_data(store_dir, key, compress=compress)
    
    if not saved_data.empty:
        merged_df = validate_duplicate_and_merge(saved_data, new_df)
        store_to_file(merged_df, key, compress=compress)
        return merged_df

    store_to_file(new_df, key, compress=compress)
    return new_df

def plot_data(df):
    fig = go.Figure(data=go.Ohlc(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],                    
    ))
    fig.show()



if __name__ == "__main__":
    from IPython.display import display
    # pass
    ticker = 'BTC-USD'
    data_store = '_data_store'

    # df = pd.read_csv(data_store+'/'+ticker+'.zip', parse_dates=True, keep_date_col=True, index_col=0)
    df = pd.read_csv(data_store+'/'+ticker+'.zip')
    
    # df = get_data(data_store, ticker, compress=True)
    display(df.tail)
