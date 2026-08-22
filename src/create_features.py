import pandas as pd

from src.features import (
    add_daily_return,
    add_momentum,
    add_volatility,
    add_price_vs_sma,
    add_volume_ratio,
)


INPUT_FILE = "data/AAPL_clean.csv"
OUTPUT_FILE = "data/AAPL_features.csv"


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0)

    # Daily return
    df = add_daily_return(df)

    # Momentum features
    df = add_momentum(df, window=5)
    df = add_momentum(df, window=10)
    df = add_momentum(df, window=20)

    # Volatility features
    df = add_volatility(df, window=5)
    df = add_volatility(df, window=10)
    df = add_volatility(df, window=20)

    # Price relative to moving averages
    df = add_price_vs_sma(df, window=10)
    df = add_price_vs_sma(df, window=20)

    # Volume
    df = add_volume_ratio(df, window=20)

    df.to_csv(OUTPUT_FILE)

    print(f"Saved features to {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLast 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    main()
