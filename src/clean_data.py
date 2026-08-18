import pandas as pd


def load_raw_data(file_path):
    data = pd.read_csv(file_path, header=[0, 1], index_col=0)

    return data


def clean_columns(data):
    data.columns = data.columns.get_level_values(0)

    return data


def main():
    data = load_raw_data("data/AAPL_raw.csv")
    data = clean_columns(data)

    data.to_csv("data/AAPL_clean.csv")

    print("Saved cleaned data to data/AAPL_clean.csv")
    print("Shape:", data.shape)


if __name__ == "__main__":
    main()
