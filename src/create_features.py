import pandas as pd

from src.features import add_daily_return


INPUT_FILE = "data/AAPL_clean.csv"
OUTPUT_FILE = "data/AAPL_features.csv"


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0)

    df = add_daily_return(df)

    df.to_csv(OUTPUT_FILE)

    print(f"Saved features to {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
