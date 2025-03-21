import talib as ta
from talib import MA_Type
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import backtesting
from backtesting import Backtest
import itertools
import logging as log
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
import numpy as np
import multiprocessing as mp


def drop_multi_index_columns(df):
    if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
        print("Drop multiindex header", type(df.columns))
        df.columns = df.columns.droplevel(0)
    return df


# rename Datetime column to Date
def rename_index_datetime_to_date(df):
    df.index.names = ["Date"]


def rename_index_date_to_datetime(df):
    df.index.names = ["Datetime"]


def strategy_01(df, current_candle):
    current_pos = df.index.get_loc(current_candle)

    c1 = df["Low"].iloc[current_pos - 4] > df["High"].iloc[current_pos]
    c2 = df["High"].iloc[current_pos] > df["Low"].iloc[current_pos - 3]
    c3 = df["Low"].iloc[current_pos - 3] > df["Low"].iloc[current_pos - 2]
    c4 = df["Low"].iloc[current_pos - 2] > df["Low"].iloc[current_pos - 1]
    c5 = df["Close"].iloc[current_pos] > df["High"].iloc[current_pos - 1]

    if c1 and c2 and c3 and c4 and c5:
        return 2

    c1 = df["High"].iloc[current_pos - 4] < df["Low"].iloc[current_pos]
    c2 = df["Low"].iloc[current_pos] < df["High"].iloc[current_pos - 3]
    c3 = df["High"].iloc[current_pos - 3] < df["High"].iloc[current_pos - 2]
    c4 = df["High"].iloc[current_pos - 2] < df["High"].iloc[current_pos - 1]
    c5 = df["Close"].iloc[current_pos] < df["Low"].iloc[current_pos - 1]

    if c1 and c2 and c3 and c4 and c5:
        return 1  # Return 1 as the function result for these symmetrical conditions

    return 0


def add_signal(df):
    df["Signal"] = df.progress_apply(lambda row: strategy_01(df, row.name), axis=1)
    return df


def add_pointpos_column(df, signal_column):
    """
    Adds a 'pointpos' column to the DataFrame to indicate the position of support and resistance points.

    Parameters:
    df (DataFrame): DataFrame containing the stock data with the specified SR (support/resistance) column, 'Low', and 'High' columns.
    sr_column (str): The name of the column to consider for the SR (support/resistance) points.

    Returns:
    DataFrame: The original DataFrame with an additional 'pointpos' column.
    """

    def pointpos(row):
        if row[signal_column] == 2:
            return row["Low"] - 1e-4
        elif row[signal_column] == 1:
            return row["High"] + 1e-4
        else:
            return np.nan

    df["pointpos"] = df.apply(lambda row: pointpos(row), axis=1)
    return df


def add_pointpos_column(df, signal_column, coef):
    def pointpos(row):
        if row[signal_column] == 2:
            return row["Low"] - coef
        elif row[signal_column] == 1:
            return row["High"] + coef
        else:
            return np.nan

    df["pointpos"] = df.apply(lambda row: pointpos(row), axis=1)
    return df


def plot_candlestick_with_signals(df, start_index, num_rows):
    """
    Plots a candlestick chart with signal points.

    Parameters:
    df (DataFrame): DataFrame containing the stock data with 'Open', 'High', 'Low', 'Close', and 'pointpos' columns.
    start_index (int): The starting index for the subset of data to plot.
    num_rows (int): The number of rows of data to plot.

    Returns:
    None
    """
    df_subset = df[start_index : start_index + num_rows]

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Candlestick(
            x=df_subset.index,
            open=df_subset["Open"],
            high=df_subset["High"],
            low=df_subset["Low"],
            close=df_subset["Close"],
            name="Candlesticks",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_subset.index,
            y=df_subset["pointpos"],
            mode="markers",
            marker=dict(size=10, color="MediumPurple", symbol="circle"),
            name="Entry Points",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        width=1200,
        height=800,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(family="sans-serif", size=12, color="white"),
            bgcolor="black",
            bordercolor="gray",
            borderwidth=2,
        ),
    )

    fig.show()


def add_bb_rsi_signal(
    df, rsi_threshold_low=30, rsi_threshold_high=70, bb_width_threshold=0.0015
):
    df["Signal"] = 0

    for i in range(1, len(df)):
        # Previous candle conditions
        prev_candle_closes_below_bb = df["Close"].iloc[i - 1] < df["bb_low"].iloc[i - 1]
        prev_rsi_below_thr = df["rsi"].iloc[i - 1] < rsi_threshold_low
        # Current candle conditions
        closes_above_prev_high = df["Close"].iloc[i] > df["High"].iloc[i - 1]
        bb_width_greater_threshold = df["bb_width"].iloc[i] > bb_width_threshold

        # Combine conditions
        if (
            prev_candle_closes_below_bb
            and prev_rsi_below_thr
            and closes_above_prev_high
            and bb_width_greater_threshold
        ):
            df.at[i, "Signal"] = 2  # Set the buy signal for the current candle

        # Previous candle conditions
        prev_candle_closes_above_bb = df["Close"].iloc[i - 1] > df["bb_up"].iloc[i - 1]
        prev_rsi_above_thr = df["rsi"].iloc[i - 1] > rsi_threshold_high
        # Current candle conditions
        closes_below_prev_low = df["Close"].iloc[i] < df["Low"].iloc[i - 1]
        bb_width_greater_threshold = df["bb_width"].iloc[i] > bb_width_threshold

        # Combine conditions
        if (
            prev_candle_closes_above_bb
            and prev_rsi_above_thr
            and closes_below_prev_low
            and bb_width_greater_threshold
        ):
            df.at[i, "Signal"] = 1  # Set the sell signal for the current candle

    return df


def strategy_02(df, current_candle):
    current_pos = df.index.get_loc(current_candle)
    c0 = df["Open"].iloc[current_pos] > df["Close"].iloc[current_pos]
    # Condition 1: The high is greater than the high of the previous day
    c1 = df["High"].iloc[current_pos] > df["High"].iloc[current_pos - 1]
    # Condition 2: The low is less than the low of the previous day
    c2 = df["Low"].iloc[current_pos] < df["Low"].iloc[current_pos - 1]
    # Condition 3: The close of the Outside Bar is less than the low of the previous day
    c3 = df["Close"].iloc[current_pos] < df["Low"].iloc[current_pos - 1]

    if c0 and c1 and c2 and c3:
        return 2  # Signal for entering a Long trade at the open of the next bar

    c0 = df["Open"].iloc[current_pos] < df["Close"].iloc[current_pos]
    # Condition 1: The high is greater than the high of the previous day
    c1 = df["Low"].iloc[current_pos] < df["Low"].iloc[current_pos - 1]
    # Condition 2: The low is less than the low of the previous day
    c2 = df["High"].iloc[current_pos] > df["High"].iloc[current_pos - 1]
    # Condition 3: The close of the Outside Bar is less than the low of the previous day
    c3 = df["Close"].iloc[current_pos] > df["High"].iloc[current_pos - 1]

    if c0 and c1 and c2 and c3:
        return 1

    return 0


def add_signal_to_strategy_02(df):
    df["Signal"] = df.progress_apply(lambda row: strategy_02(df, row.name), axis=1)
    return df


def ema_signal(df, current_candle, backcandles):
    df_slice = df.reset_index().copy()
    # Get the range of candles to consider
    start = max(0, current_candle - backcandles)
    end = current_candle
    relevant_rows = df_slice.iloc[start:end]

    if all(relevant_rows["High"] < relevant_rows["EMA"]):
        return 1
    elif all(relevant_rows["Low"] > relevant_rows["EMA"]):
        return 2
    else:
        return 0


def add_ema_signal(df):
    df["Signal"] = df.progress_apply(
        lambda row: ema_signal(df, row.name, 5) if row.name >= 20 else 0, axis=1
    )
    return df


def ema_signal2(df, current_candle, back_candles):
    df_slice = df.reset_index().copy()

    df_slice = df_slice.loc[
        current_candle - back_candles : current_candle, ["Open", "Close", "ema"]
    ]
    dnt = 0 if (df_slice[["Open", "Close"]].max(axis=1) >= df_slice["ema"]).any() else 1
    upt = 0 if (df_slice[["Open", "Close"]].min(axis=1) <= df_slice["ema"]).any() else 1

    if upt == 1 and dnt == 1:
        return 3
    elif upt == 1:
        return 2
    elif dnt == 1:
        return 1
    else:
        return 0


def add_ema200_indicators(data, params):
    atr_period = params["atr_period"]
    rsi_period = params["rsi_period"]

    df = data.copy()
    df["rsi"] = ta.RSI(df["Close"], timeperiod=rsi_period)
    df["atr"] = ta.ATR(
        low=df["Low"], close=df["Close"], high=df["High"], timeperiod=atr_period
    )
    df["ema"] = ta.EMA(df["Close"], timeperiod=200)
    return df


def add_ema200_signal(data, back_candles=8):
    df = data.copy()
    for row in range(back_candles - 1, len(df)):
        upt = 1
        dnt = 1
        for i in range(row - back_candles, row + 1):
            if df.High[row] >= df.EMA200[row]:
                dnt = 0
            if df.Low[row] <= df.EMA200[row]:
                upt = 0
        if upt == 1 and dnt == 1:
            df["EMASignal"][row] = 3  # when trend loop
        elif upt == 1:
            df["EMASignal"][row] = 2
        elif dnt == 1:
            df["EMASignal"][row] = 1

    return df


def add_signal_by_ema200(data):
    df = data.copy()
    for row in range(0, len(df)):
        if df.EMAsignal[row] == 1 and df.RSI[row] >= 90:
            df["Signal"] = 1
        if df.EMAsignal[row] == 2 and df.RSI[row] <= 10:
            df["Signal"] = 2
    return df


def add_bb_rsi_strategy(data, params):
    df = data.copy()
    # Params
    bb_period = params["bb_period"]
    atr_period = params["atr_period"]
    rsi_period = params["rsi_period"]
    ema_period = params["ema_period"]
    bb_std = params["bb_std"]

    df = df[df.High != df.Low]

    # df['bb_up'], df['bb_mid'], df['bb_low'] = ta.BBANDS(df["Close"], nbdevup=bb_std, nbdevdn=bb_std, timeperiod=bb_period)
    df["bb_up"], df["bb_mid"], df["bb_low"] = ta.BBANDS(
        df["Close"],
        timeperiod=bb_period,
        nbdevup=bb_std,
        nbdevdn=bb_std,
        matype=MA_Type.SMA,
    )
    # df['bb_up'], df['bb_mid'], df['bb_low'] = ta.BBANDS(df["Close"], timeperiod=bb_period, matype=MA_Type.EMA)
    # минимальный порог для ширины полосы BBANDS, когда поступает сигнал на вход.
    # Для избежания торговли в зонах низкой волатильности
    df["bb_width"] = (df["bb_up"] - df["bb_low"]) / df["bb_mid"]
    # Подтверждения тренда
    df["rsi"] = ta.RSI(df["Close"], timeperiod=rsi_period)
    # ATR управления торговлей
    # для подсчета стоп-лосса и тейк-профита
    df["atr"] = ta.ATR(
        low=df["Low"], close=df["Close"], high=df["High"], timeperiod=atr_period
    )
    df["ema"] = ta.EMA(df["Close"], timeperiod=ema_period)
    return df


def apply_bb_rsi_strategy(data, params):
    df = add_bb_rsi_strategy(data.copy(), params)

    bb_width_threshold = params["bb_width_threshold"]
    # Signal, 2-buy, 1-sell
    add_bb_rsi_signal(
        df=df,
        rsi_threshold_low=30,
        rsi_threshold_high=70,
        bb_width_threshold=bb_width_threshold,
    )

    return df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "bb_up",
            "bb_mid",
            "bb_low",
            "bb_width",
            "rsi",
            "atr",
            "Signal",
        ]
    ]


def apply_signal_by_ema_macd(data, params):
    df = add_bb_rsi_strategy(data.copy(), params).copy()
    back_candles = params["back_candles"]

    def add_signal(df, current_candle, back_candles):
        if (
            ema_signal(df, current_candle, back_candles) == 2
            and all(
                df.loc[current_candle - 3 : current_candle - 2, "MACD"]
                < df.loc[current_candle - 3 : current_candle - 2, "MACD_signal"]
            )
            and all(
                df.loc[current_candle - 1 : current_candle, "MACD"]
                > df.loc[current_candle - 1 : current_candle, "MACD_signal"]
            )
        ):
            return 2
        if (
            ema_signal(df, current_candle, back_candles) == 1
            and all(
                df.loc[current_candle - 3 : current_candle - 2, "MACD"]
                > df.loc[current_candle - 3 : current_candle - 2, "MACD_signal"]
            )
            and all(
                df.loc[current_candle - 1 : current_candle, "MACD"]
                < df.loc[current_candle - 1 : current_candle, "MACD_signal"]
            )
        ):

            return 1
        return 0

    df["Signal"] = df.progress_apply(
        lambda row: add_signal(df, row.name, back_candles) if row.name != 0 else 0,
        axis=1,
    )
    return df


def apply_bb_rsi_ema_strategy(data, params):
    back_candles = params["back_candles"]

    df = add_bb_rsi_strategy(data.copy(), params)

    def add_signal(df, current_candle, back_candles=7):
        if (
            ema_signal2(df, current_candle, back_candles) == 2
            and df.Close[current_candle] <= df["bb_low"][current_candle]
            # and df.RSI[current_candle]<60
        ):
            return 2
        if (
            ema_signal2(df, current_candle, back_candles) == 1
            and df.Close[current_candle] >= df["bb_up"][current_candle]
            # and df.RSI[current_candle]>40
        ):
            return 1
        return 0

    df["Signal"] = df.progress_apply(
        lambda row: add_signal(df, row.name, back_candles) if row.name != 0 else 0,
        axis=1,
    )

    return df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "bb_up",
            "bb_mid",
            "bb_low",
            "bb_width",
            "rsi",
            "atr",
            "Signal",
        ]
    ]


def apply_ema200_strategy(data, params):
    df = add_ema200_indicators(data.copy, params)
    df = add_ema200_signal(df, params)
    df = add_signal_by_ema200(df)


def backtest_bb_rsi_strategy(df, strategy_class, params):
    bt_df = df.copy()
    bt_df.rename(columns={"Date": "Datetime"}, inplace=True)
    bt_df["Datetime"] = pd.to_datetime(bt_df["Datetime"])
    bt_df.set_index("Datetime", inplace=True)
    bt_df = bt_df.dropna()

    log.debug(f"NaN: {bt_df.isnull().sum()}")

    bt = Backtest(
        bt_df,
        strategy_class,
        cash=params["cash"],
        commission=0.001,
        exclusive_orders=True,
        margin=1 / 10,
    )

    maximize = params["maximize"]
    max_tries = params["max_tries"]
    method = params["method"]
    if method == "skopt":
        stats, optimize_result = bt.optimize(
            slcoef=[i / 10 for i in range(10, 21)],  # 16
            TPcoef=[i / 10 for i in range(10, 21)],
            method=method,  #'skopt', 'grid',
            maximize=maximize,
            max_tries=max_tries,
            random_state=0,
            return_heatmap=False,
            return_optimization=True,
        )
        return stats, optimize_result
    else:
        stats = bt.optimize(
            slcoef=[i / 10 for i in range(10, 21)],  # 16
            TPcoef=[i / 10 for i in range(10, 21)],
            method=method,  #'skopt', 'grid',
            maximize=maximize,
            max_tries=max_tries,
            random_state=0,
            return_heatmap=False,
        )
        return stats


def get_backtesting_params(parameters):
    params = {
        "cash": 10_000,
        "maximize": "SQN",  # stats item
        "max_tries": 30,
        "method": "grid",  # skopt
        "stats_item": "Return [%]",
        "bb_std": 1.5,
        "back_candles": 8,
        "ema_period": 30,
    }

    if parameters.get("maximize") != None:
        params["maximize"] = parameters["maximize"]
    if parameters.get("method") != None:
        params["method"] = parameters["method"]
    if parameters.get("cash") != None:
        params["cash"] = parameters["cash"]
    if parameters.get("maximize") != None:
        params["maximize"] = parameters["maximize"]
    if parameters.get("max_tries") != None:
        params["max_tries"] = parameters["max_tries"]
    if parameters.get("stats_item") != None:
        params["stats_item"] = parameters["stats_item"]
    if parameters.get("bb_std") != None:
        params["bb_std"] = parameters["bb_std"]
    if parameters.get("back_candles") != None:
        params["back_candles"] = parameters["back_candles"]
    if parameters.get("ema_period") != None:
        params["ema_period"] = parameters["ema_period"]
    return params


def get_best_bb_rsi_strategy(train_data, strategy_class, parameters, apply_signal_func):
    bb_period_list = parameters["bb_period_list"]
    bb_width_threshold_list = parameters["bb_width_threshold_list"]
    atr_period_list = parameters["atr_period_list"]
    rsi_period_list = parameters["rsi_period_list"]

    params = get_backtesting_params(parameters)

    best_params = None
    best_performance = -float("inf")

    for bb_period, bb_width_threshold, atr_period, rsi_period in itertools.product(
        bb_period_list, bb_width_threshold_list, atr_period_list, rsi_period_list
    ):
        params["bb_period"] = bb_period
        params["bb_width_threshold"] = bb_width_threshold
        params["atr_period"] = atr_period
        params["rsi_period"] = rsi_period

        df_with_signal = apply_signal_func(train_data.copy(), params).copy()

        if params["method"] == "skopt":
            stats, optimize_result = backtest_bb_rsi_strategy(
                df_with_signal, strategy_class, params
            )
            log.debug(f"optimize_result: {optimize_result}")
        else:
            stats = backtest_bb_rsi_strategy(df_with_signal, strategy_class, params)
        log.debug(f"Stats: {stats}")

        performance = stats[params["stats_item"]]
        log.debug(
            "Performance, current: {performance}, best_performance: {best_performance}"
        )

        if performance > 0:
            print(
                f"Performance, current: {performance}, best_performance: {best_performance}, strategy_class: {strategy_class}"
            )
            best_performance = performance
            best_params = params

    log.info("Best Performance: {}".format(best_performance))
    log.info("Best Parameters: {}".format(best_params))
    return best_params, best_performance


def get_best_ema200_strategy(train_data, strategy_class, parameters):
    atr_period_list = parameters["atr_period_list"]
    rsi_period_list = parameters["rsi_period_list"]

    params = get_backtesting_params(parameters)

    best_params = None
    best_performance = -float("inf")

    for atr_period, rsi_period in itertools.product(atr_period_list, rsi_period_list):
        params["atr_period"] = atr_period
        params["rsi_period"] = rsi_period

        df_with_signal = apply_ema200_strategy(train_data.copy(), params).copy()

        if params["method"] == "skopt":
            stats, optimize_result = backtest_bb_rsi_strategy(
                df_with_signal, strategy_class, params
            )
            log.debug(f"optimize_result: {optimize_result}")
        else:
            stats = backtest_bb_rsi_strategy(df_with_signal, strategy_class, params)
        log.debug(f"Stats: {stats}")

        performance = stats[params["stats_item"]]
        log.debug(
            "Performance, current: {performance}, best_performance: {best_performance}"
        )

        if performance > 0 and performance > best_performance:
            print(
                f"Performance, current: {performance}, best_performance: {best_performance}, strategy_class: {strategy_class}"
            )
            best_performance = performance
            best_params = params

    log.info("Best Performance: {}".format(best_performance))
    log.info("Best Parameters: {}".format(best_params))
    return best_params, best_performance


def walk_forward_optimization_by_date_range(
    start_date, end_date, train_size, test_size
):
    delta = relativedelta(end_date, start_date)
    num_iterations = int((delta.years - train_size) // test_size)

    dates_list = []
    for i in range(num_iterations + 1):
        start_train = start_date + relativedelta(years=test_size * i)
        end_train = start_train + relativedelta(years=train_size, days=-1)
        start_test = end_train + relativedelta(days=+1)
        end_test = start_test + relativedelta(years=test_size, days=-1)
        if end_test > end_date:
            end_test = end_date

        # print(f"{start_train}, {end_train}, {start_test}, {end_test}")

        dates_list.append(
            {
                "train_dates": [start_train, end_train],
                "test_dates": [start_test, end_test],
            }
        )

    return dates_list


def walk_forward_optimization_by_date_range2(
    start_date, end_date, train_size_in_month, test_size_in_month
):
    delta = relativedelta(end_date, start_date)
    months_diff = (delta.years * 12) + delta.months
    num_iterations = int((months_diff - train_size_in_month) // test_size_in_month)

    dates_list = []
    for i in range(num_iterations + 1):
        start_train = start_date + relativedelta(months=test_size_in_month * i)
        end_train = start_train + relativedelta(months=train_size_in_month, days=-1)
        start_test = end_train + relativedelta(days=+1)
        end_test = start_test + relativedelta(months=test_size_in_month, days=-1)
        if end_test > end_date:
            end_test = end_date

        # print(f"{start_train}, {end_train}, {start_test}, {end_test}")

        dates_list.append(
            {
                "train_dates": [start_train, end_train],
                "test_dates": [start_test, end_test],
            }
        )

    return dates_list


def walk_forward_optimization_by_index(index_length, train_size, test_size):
    num_iterations = (index_length - train_size) // test_size

    index_list = []
    for i in range(num_iterations + 1):
        start_train = i * test_size
        end_train = start_train + train_size
        start_test = end_train
        end_test = start_test + test_size
        if end_test > index_length:
            end_test = index_length

        # print(f"{start_train}, {end_train}, {start_test}, {end_test}")

        index_list.append(
            {
                "train_indexes": [start_train, end_train],
                "test_indexes": [start_test, end_test],
            }
        )

    return index_list


def run_backtest(df, strategy_class, params):
    bt_df = df.copy()
    cash = 10_000
    commission = 0.002
    max_tries = 300
    maximize = "Return [%]"

    if params.get("cash") != None:
        cash = params["cash"]

    if params.get("commission") != None:
        commission = params["commission"]

    if params.get("maximize") != None:
        maximize = params["maximize"]

    if params.get("max_tries") != None:
        max_tries = params["max_tries"]

    bt = Backtest(
        bt_df,
        strategy_class,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        margin=1 / 10,
    )

    stats = bt.run(
        window_size=params["window_size"],
        dp=params["dp"],
        model=params["model"],
    )

    # stats = bt.optimize(
    #     maximize=maximize,
    #     max_tries=max_tries,
    #     random_state=0,
    #     return_heatmap=False,
    #     window_size=params["window_size"],
    #     dp=params["dp"],
    #     model=params["model"],
    # )

    return bt, stats
