import pandas as pd


def add_daily_return(df):
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    return df
