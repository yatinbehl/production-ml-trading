import pandas as pd


def add_daily_return(df):
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    return df


def add_momentum(df, window=5):
    df = df.copy()
    df[f"momentum_{window}d"] = df["Close"].pct_change(periods=window)
    return df


def add_volatility(df, window=5):
    df = df.copy()
    df[f"volatility_{window}d"] = df["daily_return"].rolling(window=window).std()
    return df
