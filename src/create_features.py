import pandas as pd

from src.features import (
    add_daily_return,
    add_momentum,
    add_volatility,
)


INPUT_FILE = "data/AAPL_clean.csv"
OUTPUT_FILE = "data/AAPL_features.csv"


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0)

    df = add_daily_return(df)
    df = add_momentum(df, window=5)
    df = add_volatility(df, window=5)

    df.to_csv(OUTPUT_FILE)

    print(f"Saved features to {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLast 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    main()
