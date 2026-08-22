import pandas as pd

from src.target import add_target


INPUT_FILE = "data/AAPL_features.csv"
OUTPUT_FILE = "data/AAPL_ml.csv"


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0)

    df = add_target(df, horizon=5)

    df.to_csv(OUTPUT_FILE)

    print(f"Saved ML dataset to {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLast 10 rows:")
    print(df.tail(10))


if __name__ == "__main__":
    main()
