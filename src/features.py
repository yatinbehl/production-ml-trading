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
def add_price_vs_sma(df, window=20):
    df = df.copy()

    sma = df["Close"].rolling(window=window).mean()

    df[f"price_vs_sma_{window}d"] = (
        df["Close"] / sma - 1
    )

    return df

def add_volume_ratio(df, window=20):
    df = df.copy()

    average_volume = df["Volume"].rolling(window=window).mean()

    df[f"volume_ratio_{window}d"] = (
        df["Volume"] / average_volume
    )

    return df
