import pandas as pd


def add_target(df, horizon=5, threshold=0.01):
    df = df.copy()

    df["future_close"] = df["Close"].shift(-horizon)

    df["future_return"] = (
        df["future_close"] / df["Close"] - 1
    )

    df["target"] = pd.NA

    valid_rows = df["future_return"].notna()

    df.loc[valid_rows, "target"] = (
        df.loc[valid_rows, "future_return"] > threshold
    ).astype(int)

    return df
