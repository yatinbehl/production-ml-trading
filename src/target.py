import pandas as pd


def add_target(df, horizon=5):
    df = df.copy()

    df["future_close"] = df["Close"].shift(-horizon)

    df["target"] = pd.NA

    valid_rows = df["future_close"].notna()

    df.loc[valid_rows, "target"] = (
        df.loc[valid_rows, "future_close"]
        > df.loc[valid_rows, "Close"]
    ).astype(int)

    return df
